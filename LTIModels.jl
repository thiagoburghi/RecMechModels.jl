"""
    Orthogonal Filterbank cell. Creates a state-space model 
        x(t+1) = A x(t) + B u(t)
        y(t) = x(t)
    where the impulse responses of the states are mutually orthonormal.
    (checked in 20/04/2023 that this is working properly.)

    Constructor takes τ, an array containing arbitrary continuous-time time-constants .

    NB: When the poles of the GOBF transfer functions are all distinct, the
    neural net could learn to undo the realization we selected, and in this 
    case the realization only matters as a form of initial condition or prior.
    Indeed, if the GOBF poles are distinct, then there is a transformation T 
    whose columns are the eigenvectors of A^gobf such that Tx^diag(t) = x^gobf(t)
    Then if we select a GOBF realization, the neural network's input weight matrix
    (assuming for simplicity that the number of units is = number of filter banks)
    could be learned as W = T⁻¹ so that the input to the first activation functions
    is x^diag(t). In practice I haven't seen much difference between the realizations.
"""
struct OrthogonalFilterCell{M<:AbstractMatrix} <: AbstractLTI
    W::M
    B::M
    state0::M
    trainable::Bool
    dt::Float32
end

function OrthogonalFilterCell(τ::V,dt;trainable=false) where {V<:AbstractVector}
    λ = [exp.(-dt./τᵢ) for τᵢ in τ]
    W,B = GOBFRealization(λ)
    W,B,state0 = map(Matrix{Float32},(W,B,zeros(size(W,1),1)))
    return OrthogonalFilterCell(W,B,state0,trainable,Float32(dt))
end

function reduceLTI(m::OrthogonalFilterCell,inds::AbstractVector{Int})
    return OrthogonalFilterCell(m.W[inds,inds],m.B[inds,:],m.state0[inds,:],m.trainable,m.dt)
end

function (fb::OrthogonalFilterCell)(x, u)
    x₊ = fb.W * x .+ fb.B * u
    return x₊, x # second argument y = x is the output of the filter
end

function DCgain(fb::OrthogonalFilterCell)
    return inv(Matrix(I,size(fb.W))-fb.W)*fb.B
end

function transferFunction(fb::OrthogonalFilterCell,ω::F,dt::G) where {F<:Real,G<:Real}
    return inv(exp(im*ω*dt)*Matrix(I,size(fb.W))-fb.W)*fb.B
end

function getLargestTimescale(fb::OrthogonalFilterCell)
    return maximum(eigmax(fb.W))
end

function getTimeConstants(fb::OrthogonalFilterCell)
    a,_ = eigen(fb.W)
    τ = -fb.dt ./ log.(a)
    return τ
end

Flux.@layer :expand OrthogonalFilterCell

"""
    Version of orthogonal where we can train time constants
"""
struct TrainableOrthogonalFilterCell{M<:AbstractMatrix} <: AbstractLTI
    W::M
    A::M
    B::M
    κ::Float32
    state0::M
    trainable::Bool
    dt::Float32
end

function TrainableOrthogonalFilterCell(τ::V,dt;κ=0.1,trainable=true) where {V<:AbstractVector}
    λ = [exp.(-dt./τᵢ) for τᵢ in τ]
    W,B = GOBFRealization(λ)
    A = tril(W,-1)
    W = diag(W)
    W = log.(W./(1 .- W))/κ #κ small makes the logistic shallower, allowing faster learning
    W,A,B,state0 = map(Matrix{Float32},(reshape(W,length(W),1),A,B,zeros(size(W,1),1)))
    return TrainableOrthogonalFilterCell(W,A,B,Float32(κ),state0,trainable,Float32(dt))
end

function (fb::TrainableOrthogonalFilterCell)(x, u)
    x₊ = fb.A * x .+ σ.(fb.κ*fb.W) .* x .+ fb.B * u
    return x₊, x # second argument y = x is the output of the filter
end

function DCgain(fb::TrainableOrthogonalFilterCell)
    return inv(Matrix(I,size(fb.A))-(fb.A + Diagonal(σ.(fb.κ*fb.W[:]))))*fb.B
end

function transferFunction(fb::TrainableOrthogonalFilterCell,ω::F,dt::G) where {F<:Real,G<:Real}
    return inv(exp(im*ω*dt)*Matrix(I,size(fb.A))-(fb.A + Diagonal(σ.(fb.κ*fb.W))))*fb.B
end

function getLargestTimescale(fb::TrainableOrthogonalFilterCell)
    return maximum(σ.(fb.κ*fb.W[:]))
end

function getTimeConstants(fb::TrainableOrthogonalFilterCell)
    a = σ.(fb.κ*fb.W[:])
    τ = -fb.dt ./ log.(a)
    return τ
end

Flux.@layer :expand TrainableOrthogonalFilterCell

function (fb::AbstractLTI)(u::AbstractArray;DC=true)
    DC ? state0 = DCgain(fb)*u[1] : state0 = fb.state0
    fb = Flux.Recur(fb,state0)
    return hcat([fb(uᵢ) for uᵢ in u]...)
end

function warmUp(fb::AbstractLTI,u::AbstractArray)
    # Warm up filter
    fb = Flux.Recur(fb,DCgain(fb)*u[1])
    _ = [fb(uᵢ) for uᵢ in u]
    # Run filter again, after warmUp there should be no large transients
    return hcat([fb(uᵢ) for uᵢ in u]...)
end

function Flux.trainable(fb::F) where F<:AbstractLTI
    if fb.trainable
        return (W=fb.W,)
    else
        return ()
    end
end

OrthogonalFilter(λ::Vector{AbstractFloat},dt) = Recur(OrthogonalFilterCell(λ,dt))
Recur(fb::OrthogonalFilterCell) = Flux.Recur(fb, fb.state0)

"""
    Diagonal Filterbank cell. Creates a state-space model 
    x(t+1) = diag(Λ) x(t) + (I-Λ) u(t)
    y(t) = x(t)
    where the impulse responses of the states are mutually orthonormal; here 
    Λ is a vector of time constants.

    Constructor takes τ, an array containing continuous-time time constants.
"""
struct DiagonalFilterCell{M<:AbstractMatrix} <: AbstractLTI
    W::M    #Λ for no pole constraints
    O::M
    state0::M
    trainable::Bool
    dt::Float32
end

function DiagonalFilterCell(τ::V,dt;trainable=false) where {V<:AbstractVector}
    if V <: Vector{V} where V<:AbstractVector
        τ = vcat(τ...)
    end
    λ = exp.(-dt./τ)        # convert to discrete-time pole
    w = log.(λ./(1 .- λ))   # variable transformation
    W = reshape(w,(length(w),1))
    O = ones(length(w),1)

    W,O,state0 = map(Matrix{Float32},(W,O,zeros(length(w),1)))
    return DiagonalFilterCell(W,O,state0,trainable,Float32(dt))
end

function reduceLTI(m::DiagonalFilterCell,inds::AbstractVector{Int})
    return DiagonalFilterCell(m.W[inds,:],m.O[inds,:],m.state0[inds,:],m.trainable,m.dt)
end

function (fb::DiagonalFilterCell)(x, u)
    x₊ =  σ.(fb.W) .* (x .- u) .+ fb.O * u
    return x₊, x # second argument y = x is the output of the filter
end

function OrthogonalTransformation(fb::DiagonalFilterCell;inds=nothing)
    inds = isnothing(inds) ? (1:size(fb.W,1)) : inds
    W,B = GOBFRealization(Float64.(σ.(fb.W[inds])))
    eig = eigen(W)
    T = eig.vectors
    # aux = (T \ B) ./ (1f0 .- eig.values)
    # aux = (inv(T)* B) ./ (1f0 .- eig.values)
    # aux = ((T*Diagonal(1 .- eig.values)) \ B)
    aux = (inv(T*Diagonal(1 .- eig.values)) * B)
    return Float32.(T*Diagonal(aux[:]))
end

function DCgain(fb::DiagonalFilterCell)
    return ones(Float32,size(fb.O))
end

function transferFunction(fb::DiagonalFilterCell,ω::F,dt::G) where {F<:Real,G<:Real}
    return inv(exp(im*ω*dt)*Matrix(I,size(fb.W,1),size(fb.W,1))-Diagonal(σ.(fb.W[:])))*(fb.O-σ.(fb.W))
end

function getLargestTimescale(fb::DiagonalFilterCell)
    return maximum(σ.(fb.W))
end

function getTimeConstants(fb::DiagonalFilterCell)
    τ = -fb.dt ./ log.(σ.(fb.W))
    return τ
end

Flux.@layer :expand DiagonalFilterCell
DiagonalFilter(τ::Vector{AbstractFloat},dt) = Recur(DiagonalFilterCell(τ,dt))
Recur(fb::DiagonalFilterCell) = Flux.Recur(fb, fb.state0)

"""
    Mixed Filterbank cell.
"""
struct MixedFilterCell{M<:AbstractMatrix} <: AbstractLTI
    W::M    #Λ for no pole constraints
    O::M
    A::M
    B::M
    state0::M
    trainable::Bool
    dt::Float32
end

function MixedFilterCell(τ::AbstractVector,dt;trainable=false)
    τdiag = τ[1]
    τgobf = τ[2:end]
    
    λdiag = exp.(-dt./τdiag)
    w = log.(λdiag./(1 .- λdiag))
    W = reshape(w,(length(w),1))
    O = ones(length(w),1)

    λgobf = [exp.(-dt./τᵢ) for τᵢ in τgobf] # exp.(-dt./τgobf) # 
    A,B = GOBFRealization(λgobf)

    W,O,A,B,state0 = map(Matrix{Float32},(W,O,A,B,zeros(length(O)+length(B),1)))
    return MixedFilterCell(W,O,A,B,state0,trainable,Float32(dt))
end

function (fb::MixedFilterCell)(x, u)
    Λ = σ.(fb.W)
    xdiag₊ =  Λ .* x[1:size(fb.W,1),:] .+ (fb.O .- Λ) * u
    xgobf₊ =  fb.A * x[size(fb.W,1)+1:end,:] .+ fb.B * u
    return vcat(xdiag₊,xgobf₊), x # second argument y = x is the output of the filter
end

function DCgain(fb::MixedFilterCell)
    DCdiag = fb.O
    DCgobf = inv(Matrix(I,size(fb.A))-fb.A)*fb.B
    return vcat(DCdiag,DCgobf)
end

function transferFunction(fb::MixedFilterCell,ω::F,dt::G) where {F<:Real,G<:Real}
    TFdiag = inv(exp(im*ω*dt)*Matrix(I,size(fb.W,1),size(fb.W,1))-Diagonal(σ.(fb.W[:])))*(fb.O-σ.(fb.W))
    TFgobf = inv(exp(im*ω*dt)*Matrix(I,size(fb.A))-fb.A)*fb.B
    return vcat(TFdiag,TFgobf)
end

function getLargestTimescale(fb::MixedFilterCell)
    return maximum([maximum(σ.(fb.W));eigen(fb.A).values])
end

function getTimeConstants(fb::MixedFilterCell)
    τdiag = -fb.dt ./ log.(σ.(fb.W))
    τgobf = -fb.dt ./ log.(eigen(fb.A).values)
    return vcat(τdiag,τgobf)
end

Flux.@layer :expand MixedFilterCell
MixedFilter(τ::Vector{AbstractFloat},dt) = Recur(MixedFilterCell(τ,dt))
Recur(fb::MixedFilterCell) = Flux.Recur(fb, fb.state0)