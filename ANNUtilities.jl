"""
    MLP:
    A chain of specified layers with given activation functions.
"""
function MLP(hp::MlpHP,rng::AbstractRNG)
    layerUnits = (hp.nInputs, hp.layerUnits...)
    layerBiases = push!([true for i = 1:length(hp.layerUnits)-1],hp.readoutBias)
    layers = [hp.layerTypes[i](layerUnits[i]=>layerUnits[i+1],layerBiases[i],hp.layerFun[i],rng) for i = 1:length(hp.layerTypes)]
    return Chain(layers...)
end

function initPositive(nInputs,nOutputs,fun,rng;gain=1.0)
    # r = rand(rng,nOutputs,nInputs)
    r = fill(0.5,nOutputs,nInputs) + (rand(rng,nOutputs,nInputs).-0.5)/2
    if nOutputs > 1
        W = gain*4/nInputs*r
        b = collect(LinRange(-1,1,nOutputs))
    else
        if nInputs==1
            W = gain*r
        else
            W = gain*4/nInputs*r
        end
        b = [0.0]
    end
    return scalepars(W,b,fun)
end

function initGlorot(nInputs,nOutputs,fun,rng;gain=1.0,initbias::Int=0)
    r = (2-abs(initbias))*(rand(rng,nOutputs,nInputs) .- 0.5) .+ initbias/2
    if nOutputs > 1
        W = gain*sqrt(6/(nInputs+nOutputs))*r
        b = collect(LinRange(-1,1,nOutputs))
    else
        if nInputs==1
            W = gain*r
        else
            W = gain*sqrt(6/(nInputs+nOutputs))*r
        end
        b = [0.0]
    end
    return scalepars(W,b,fun)
end

"""
    Nguyen-widron initialization. Random parameters are initialized 
    assuming that the inputs to each layer are between -1 and 1 
    (these are either the range of the previous activation functions, 
    or the range of the inputs).
"""
function initNW(nInputs,nOutputs,fun,rng)
    W = 2*rand(rng,nOutputs,nInputs) .- 1
    for i = 1:size(W,1)
        W[i,:] = W[i,:]/sqrt(sum(W[i,:].^2))
    end
    
    if nOutputs > 1
        b = collect(LinRange(-1,1,nOutputs)) #.*sign.(W[:,1]) this is key to discover vhalfs
    else
        b = [0.0]
    end

    m = 0.7*nOutputs^(1/nInputs)    #0.7
    return scalepars(m*W,m*b,fun)
end

# The scalepars function modifies the weights and biases of the layer to ensure
# that W*x+b will be within the range of the layer's activation function a. 
# It assumes that the output of the previous activation function is between -1 and 1.
function scalepars(W,b,a)
    if a == tanh
        A = [-2,2]
    elseif a == slantedTanh     # for tanh(x) + 0.1f0*x
        A = [-1.3235,1.3235]
    elseif a == identity
        A = [-1,1]
    elseif a == σ
        A = [-4,4]
    else    
        A = [0,6]
    end
    x=0.5*(A[2]-A[1])   # the length of the activation function's range
    y=0.5*(A[2]+A[1])   # the average (half-activation) of the activation range
    W = x*W
    b = x*b .+ y        # moves biases to the middle of the activation range

    # Convert previous layer output range to [-1,1] 
    # (not necessary for tansig)
    # α and β are the min and max of the previous layer's output
    # x = 2/(β-α)
    # y = -(β+α)/(β-α)
    # b = w*y+b
    # w = w*x

    return (W,b)
end

function mapMinMax(X,ymin,ymax)
    # Finds the linear transformation that projects
    # each row of X onto the interval [ymin,ymax].
    xmax = maximum(X,dims=2)
    xmin = minimum(X,dims=2)
    
    # the transformation is
    # Y = ymin .+ (ymax-ymin)*(X .- xmin)./(xmax-xmin)
    # Which means each row y of Y is given by
    # y = A*x + b
    # where x is a row of X and
    D = vec((ymax-ymin) ./ (xmax-xmin))
    off = vec(ymin .- (ymax-ymin)*xmin ./ (xmax-xmin))

    # return x -> A*x .+ b
    return D,off
end

function centerNorm(X,ymin,ymax)
    μ = mean(X,dims=2)
    X = X .- μ
    xmax = maximum(X,dims=2)
    xmin = minimum(X,dims=2)

    D = zeros(size(X,1))
    for i=1:size(X,1)
        if abs(xmax[i]) > abs(xmin[i])
            D[i] = abs(ymax)/abs(xmax[i])
        else
            D[i] = abs(ymin)/abs(xmin[i])
        end
    end 

    return D,vec(-D.*μ)
end

"""
    Constant Scaling
"""
struct ConstantScaling <: AbstractLayer
    weight::Float32
end
function ConstantScaling(weight::Real)
    return ConstantScaling(Float32(weight))
end
(m::ConstantScaling)(x::AbstractVecOrMat) = m.weight*sum(x,dims=1)
weight(m::ConstantScaling) = m.weight

"""
    NormLayer:
    layer used to implement custom normalization layers.
"""
struct NormLayer{V1<:AbstractVector,V2<:Union{Nothing,AbstractVector}} <: AbstractLayer
    D::V1
    b::V2
end

function NormLayer(D::AbstractVector)
    return NormLayer(D,nothing)
end

function MinMaxNorm(input,min,max)
    D,b=mapMinMax(input,min,max)
    return NormLayer(Float32.(D),Float32.(b))
end

function CenterNorm(input,min,max)
    D,b=centerNorm(input,min,max)
    return NormLayer(Float32.(D),Float32.(b))
end

(m::NormLayer{V1,V2})(x::AbstractVecOrMat) where {V1,V2<:AbstractVector}= m.D .* x .+ m.b
(m::NormLayer{V,Nothing})(x::AbstractVecOrMat) where V = m.D .* x

inverse(m::NormLayer{V1,V2},x::Union{Number,AbstractVecOrMat}) where {V1,V2<:AbstractVector} = (x .- m.b)./m.D
inverse(m::NormLayer{V,Nothing},x::Union{Number,AbstractVecOrMat}) where V = x ./m.D
inverse(f::typeof(identity),x::Union{Number,AbstractVecOrMat}) = x

Flux.@layer :expand NormLayer trainable=()

"""
    Selector
    A layer that takes a matrix and returns a subset of its rows
"""
struct Selector
    inds::Vector{Int}
end
(m::Selector)(x::AbstractMatrix) = x[m.inds,:]

# this edge case of Parallel (when there is a single layer) isn't currently
# treated by FLux
function (m::Parallel)(x::Tuple{Tuple})
    m.connection(m.layers[1](x[1]))
end

Flux.@layer :expand Selector trainable=()