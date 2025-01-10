adjustableSoftPlus(x,w) = Flux.softplus(w.*x)./w
slantedTanh(x) = tanh(x) + 0.1f0*x
"""
    Reversal Potential Layer
"""
struct Reversal{M<:AbstractMatrix,N<:Union{Nothing,AbstractFloat}} <: AbstractLayer
    E::M
    Eₚ::N                   # Prior on the Nernst Potential
    trainable::Bool
end

function Reversal(E::AbstractArray; trainable=true)
    return Reversal(Float32.(reshape(E,(length(E),1))),nothing,trainable)
end

function Reversal(E::AbstractFloat, dim::Int; trainable=true)
    E = E*ones(dim,1)
    return Reversal(Float32.(E),nothing,trainable)
end

function Reversal(Eprior::AbstractFloat, dim::Int, σ::Union{Nothing,AbstractFloat}; regularize=true, trainable=true, rng=Random.GLOBAL_RNG)
    E = Eprior*ones(dim,1) .+ (isnothing(σ) ? 0.0 : σ*randn(rng,dim))
    if regularize && trainable
        return Reversal(Float32.(E),Float32.(Eprior),trainable)
    else
        return Reversal(Float32.(E),nothing,trainable)
    end
end

function reversals(r::Reversal)
    return r.E
end

function regularizer(r::Reversal{M,Nothing},fun::F) where {M,F}
    return 0
end

# This should always be L2!
function regularizer(r::Reversal{M,N},fun::F) where {M,N<:AbstractFloat,F}
    return l2norm(r.E .- r.Eₚ)
end

(r::Reversal)(x::AbstractVecOrMat) = x .- r.E

Flux.@layer :expand Reversal

function Flux.trainable(r::Reversal)
    if r.trainable
        return (E=r.E,)
    else
        return ()
    end
end

"""
    Threshold layer (not really used)
"""
struct Threshold{M<:AbstractMatrix,B} <: AbstractLayer
    weight::M
    b::B
    trainable::Bool
    function Threshold(W::M,bias,trainable::Bool) where M<:AbstractMatrix
        b = Flux.create_bias(W, bias, size(W)...)
        return new{M,typeof(b)}(W,b,trainable)
    end
end

Threshold(w::AbstractMatrix;bias=false,trainable=true) = Threshold(Float32.(w),bias,trainable)

(m::Threshold)(x::AbstractVecOrMat) = adjustableSoftPlus(x.+m.b,m.weight).-m.b
weight(m::Threshold) = m.weight #Flux.softplus(m.weight)

Flux.@layer :expand Threshold
function Flux.trainable(m::Threshold)
    return m.trainable ? (weight=m.weight,b=m.b) : ()
end

"""
    Nonnegative layer (elementwise product with nonnegative weights)
"""
struct Nonnegative{M<:AbstractMatrix,B} <: AbstractLayer
    weight::M
    b::B
    trainable::Bool
    function Nonnegative(W::M,bias,trainable::Bool) where {M<:AbstractMatrix}
        b = Flux.create_bias(W, bias, size(W)...)
        return new{M,typeof(b)}(W,b,trainable)
    end
end

# Initialize the weights as the inverse of softplus
Nonnegative(w::AbstractMatrix;bias=false,trainable=true) = Nonnegative(Float32.(abs.(w) .+ log.(1 .- exp.(-abs.(w)))),bias,trainable)
Nonnegative(w::AbstractVector;kwargs...) = Nonnegative(reshape(w,length(w),1); kwargs...)
Nonnegative(w::AbstractFloat;kwargs...) = Nonnegative([w;;]; kwargs...)
Nonnegative(rng::AbstractRNG,dim;kwargs...) = Nonnegative(rand(rng,Float32,dim,1); kwargs...)

(m::Nonnegative)(x::AbstractVecOrMat) = Flux.softplus(m.weight) .* x .+ m.b
weight(m::Nonnegative) = Flux.softplus(m.weight)

function setWeight(m::Nonnegative,w::AbstractMatrix)
    m.weight .= Float32.(abs.(w) .+ log.(1 .- exp.(-abs.(w))))
end

Flux.@layer Nonnegative
function Flux.trainable(m::Nonnegative)
    if m.trainable
        if typeof(m.b)!=Bool
            return (weight=m.weight,b=m.b)
        else
            return (weight=m.weight,)
        end
    else
        return ()
    end
end

"""
    SignDefinite layer 
    A copy of Dense with elementwise positive or negative weight matrix
"""
struct SignDefinite{F, M<:AbstractMatrix, B} <: AbstractLayer
    weight::M
    bias::B
    σ::F
    sign::Int
    function SignDefinite(W::M, bias, σ::F, sign::Int) where {M<:AbstractMatrix,F}
        if sign != 1 && sign != -1
            throw(ArgumentError("sign must be either 1 or -1"))
        end
        b = Flux.create_bias(W, bias, size(W,1))
        new{F,M,typeof(b)}(W, b, σ, sign)
    end
end

# Initialize the weights as the inverse of softplus
SignDefinite(w::AbstractMatrix,bias,σ;sign) = SignDefinite(Float32.(abs.(w) .+ log.(1 .- exp.(-abs.(w)))),bias,σ,sign)
SignDefinite(w::AbstractVector,bias,σ;sign) = SignDefinite(reshape(w,1,length(w)),bias,σ,sign=sign)
SignDefinite(w::AbstractFloat,bias,σ;sign)  = SignDefinite([w;;],bias,σ,sign=sign)
function SignDefinite((n_in, n_out)::Pair{<:Integer, <:Integer},bias,σ,rng;sign)
    W,b = initPositive(n_in,n_out,σ,rng,gain=1.0)   #0.7
    # W,b = initGlorot(n_in,n_out,σ,rng,gain=0.5)
    # W,b = initNW(n_in,n_out,σ,rng) 
    bias == true ? bias = b : nothing
    return SignDefinite(W,bias,σ,sign=sign)
end

Positive(w,bias=false,σ=identity) = SignDefinite(w,bias,σ,sign=1)
Positive((n_in, n_out)::Pair{Int, Int},bias,σ,rng) = SignDefinite(n_in=>n_out,bias,σ,rng,sign=1)

Negative(w,bias=false,σ=identity) = SignDefinite(w,bias,σ,sign=-1)
Negative((n_in, n_out)::Pair{Int, Int},bias,σ,rng) = SignDefinite(n_in=>n_out,bias,σ,rng,sign=-1)

function (m::SignDefinite)(x::AbstractVecOrMat)
    weight = m.sign*Flux.softplus(m.weight)
    # f = NNlib.fast_act(m.σ, x)
    return m.σ.(weight * x .+ m.bias)
end

function setWeight(m::SignDefinite,w::AbstractMatrix)
    m.weight .= abs.(w) .+ log.(1 .- exp.(-abs.(w)))
end

Flux.@layer :expand SignDefinite
weight(m::SignDefinite) = m.sign*Flux.softplus(m.weight)

"""
    Some things to make Dense work smoothly in our framework
"""
function Flux.Dense((n_in, n_out)::Pair{<:Integer, <:Integer},bias,σ,rng)
    W,b = initNW(n_in,n_out,σ,rng)
    bias == true ? bias = b : nothing
    return Dense(Float32.(W),bias,σ)
end
function InitPositiveDense((n_in, n_out)::Pair{<:Integer, <:Integer},bias,σ,rng)
    W,b = initGlorot(n_in,n_out,σ,rng,initbias=1)
    bias == true ? bias = b : nothing
    return Dense(Float32.(W),bias,σ)
end
function InitNegativeDense((n_in, n_out)::Pair{<:Integer, <:Integer},bias,σ,rng)
    W,b = initGlorot(n_in,n_out,σ,rng,initbias=-1)
    bias == true ? bias = b : nothing
    return Dense(Float32.(W),bias,σ)
end
function setWeight(m::Dense,w::AbstractMatrix)
    m.weight .= w
end
weight(m::Dense) = m.weight