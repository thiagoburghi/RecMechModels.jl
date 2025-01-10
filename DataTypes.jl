"""
    Abstract types
"""
abstract type AbstractCurrentHP end
abstract type AbstractLTI end
abstract type AbstractCurrent end
abstract type AbstractLayer end
abstract type AbstractLoss end
abstract type AbstractData end

# For convenience, as things were initially coded as arrays
function Base.getindex(X::Tuple, i::Int, j::Int)
    return X[i][j]
end

# Also for convenience, to align syntax to Flux.jl
function Base.getindex(m::A, sym::Symbol) where A<:AbstractCurrent
    return getfield(m,sym)
end

"""
    Type with input-output data (the raw measured data):
        - V is a tuple with voltage trajectories of all neurons
        - I is a tuple with applied current trajectories for all neurons
        - T is the temperature trajectory (if it is available)
        - t is the time vector
"""
struct IOData{N,M<:AbstractMatrix,TypeI<:Tuple,TypeT<:Union{Nothing,M},A<:AbstractVector}
    V::NTuple{N, M}              
    I::TypeI
    T::TypeT
    t::A
    dt::Float32
end

# Convert to matrix in case of vectors
function IOData(V::Tuple{Vararg{A}},I::Tuple{Vararg{Union{A,Nothing}}},T::Union{A,Nothing},t::AbstractVector,dt::AbstractFloat) where A<:AbstractVector
    V = Tuple([reshape(V[i],1,length(V[i])) for i=1:length(V)])
    I = Tuple([(isnothing(I[i]) ? nothing : reshape(I[i],1,length(I[i]))) for i=1:length(I)])
    T = isnothing(T) ? nothing : reshape(T,1,length(T))
    return IOData(V,I,T,t,Float32(dt))
end

# Merge datasets (assumes all data vectors have the same number of voltages and ionicCurrents)
function IOData(data_vector::Vector{IOData},dt::AbstractFloat)
    V = Tuple([hcat([d.V[i] for d in data_vector]...) for i=1:length(data_vector[1].V)])
    I = Tuple([hcat([d.I[i] for d in data_vector]...) for i=1:length(data_vector[1].V)])
    T = isnothing(data_vector[1].T) ? nothing : hcat([d.T for d in data_vector]...)
    t = vcat([d.t[:] for d in data_vector])
    return IOData(V,I,T,t,Float32(dt))
end

function upsample(d::IOData, factor::Int)
    println("Upsampling data.")
    dt = d.dt/factor
    V = Tuple([repeat(d.V[i],inner=(1,factor)) for i=1:length(d.V)])
    I = Tuple([repeat(d.I[i],inner=(1,factor)) for i=1:length(d.I)])
    T = isnothing(d.T) ? nothing : repeat(d.T,inner=(1,factor))
    t = d.t[1]:dt:d.t[end]
    return IOData(V,I,T,t,Float32(dt))
end

function samplingFactor(data_dt::AbstractFloat, model_dt::AbstractFloat)
    if model_dt < data_dt
        println("Warning: Model sampling period is smaller than i/o data sampling period.")
        if mod(data_dt,model_dt) == 0 
            return Int(data_dt/model_dt)
        else 
            throw(ArgumentError("I/O data sampling period is not a multiple of model sampling period."))
        end
    elseif model_dt > data_dt
        throw(ArgumentError("Filter bank sampling period is larger than i/o data sampling period."))
    end
    return 1
end

Flux.@layer IOData trainable=()

"""
    Type with inputs to TotalCurrent.
"""
struct TCData{M<:AbstractMatrix,TypeT<:Union{M,Nothing}}
    V::M
    U::M
    T::TypeT
    dt::Float32
end

function TCData(d::IOData,i,j)
    return TCData(d.V[i],d.V[j],d.T,d.dt)
end


"""
    Type with data for open-loop (teacher forcing) training and validation of the ANN:
        - V is a tuple with voltage trajectories of all neurons
        - I is a tuple with applied current trajectories for all neurons
        - T is the temperature trajectory (if it is available)
        - t is the time vector
        - dV is a tuple with the time derivatives of the voltage trajectories
        - X is a matrix with precomputed filter bank outputs
"""
struct ANNData{M<:AbstractMatrix,N<:AbstractVector,O<:Union{AbstractMatrix,Nothing}}
    V::Vector{M}                  
    X::Matrix{M}
    I::N
    T::O
    dV::Vector{M}
end

Flux.@layer ANNData trainable=()

"""
    Type with state-space data:
        - V is a vector with voltage trajectories of all neurons which are estimated
        - X is a matrix with precomputed filter bank outputs
        - U is a vector with voltage trajectories of all neurons which are not estimated
        - I is a tuple with applied current trajectories of all neurons which are estimated
        - T is the temperature trajectory (if it is available)
""" 
struct SSData{TV<:Tuple,TX<:Tuple,TypeU<:Union{Tuple,Nothing},TypeI<:Tuple,TypeT<:Union{AbstractArray,Nothing}} <: AbstractData
    V::TV
    X::TX
    U::TypeU
    I::TypeI
    T::TypeT
    length::Int
    dt::Float32
end

Flux.@layer SSData trainable=()

""" 
    Initial condition type (for multiple shooting)
"""
struct InitialCondition{M<:AbstractMatrix}
    value::M
end

Flux.@layer InitialCondition

"""
    Type with state-space multiple shooting data:
        ⋅ shotsize is the size of each interval of the dataset (a shot) where the network is simulated.
        ⋅ Vseq, Useq, Iseq, dVseq is a shotsize-element Vector such that 
            Iseq[i][n] = [Iₙ(i) Iₙ(i+shotsize) Iₙ(i+2*shotsize) ... Iₙ(i+(nshots-1)*shotsize)]
          that is, it concatenates the ith element of every shot of the nth neuron input.
        ⋅ V₀ and X₀ are the RNNs initial conditions at the beginning of each shot.
    The data is defined in this way so that the RNN can be run in parallel on each one of the shots.
"""
struct MSData{V<:Tuple,U<:Tuple,I<:Tuple,T<:Tuple,V0<:Tuple,X0<:Tuple,S<:Tuple,R<:Tuple} <: AbstractData
    Vseq::V
    Useq::U
    Iseq::I
    Tseq::T
    V₀::V0
    X₀::X0
    shotsize::Int                   # size of each simulation shot
    nshots::S                       # number of shots in each dataset
    rawdata::R                      # original contiguous datasets 
    train_ic::Bool                  # should initial conditions be trained?
    dt::Float32                     # sampling period of the data
    samplingFactor::Int             # ratio of model and data sampling periods
end

function Flux.gpu(d::MSData)
    Useq = (eltype(d.Useq) == Nothing ? d.Useq : gpu(d.Useq))
    Tseq = (eltype(d.Tseq) == Nothing ? d.Tseq : gpu(d.Tseq))
    Iseq = Tuple([(eltype(d.Iseq[k]) == Nothing ? d.Iseq[k] : gpu(d.Iseq[k])) for k=1:d.shotsize])
    return MSData(gpu(d.Vseq),Useq,Iseq,Tseq,gpu(d.V₀),gpu(d.X₀),d.shotsize,d.nshots,d.rawdata,d.train_ic,Float32(d.dt),d.samplingFactor)  # Return a new struct with the modified field
end

function Flux.trainable(d::MSData)
    if d.train_ic
        return (V₀=d.V₀,X₀=d.X₀)
    else
        return ()
    end
end

Flux.@layer :expand MSData

""" 
    Type used to train the RNN with mini-batches
"""
struct RNNBatches{D<:AbstractData,R<:AbstractRNG} <: AbstractData
    batches::Vector{D}
    indices::Vector{Int}
    length::Int
    rng::R
end

function RNNBatches(batches::Vector{D}; rng=Random.GLOBAL_RNG) where {D<:AbstractData}
    RNNBatches(batches,[i for i=1:length(batches)],length(batches),rng)
end

function Base.length(d::RNNBatches)
    return length(d.batches)
end

function Base.iterate(d::RNNBatches, state=1)
    if state > length(d.batches)
        for i = 1:length(d.batches)
            push!(d.indices,i)
        end
        return nothing
    else
        random_index = rand(d.rng,1:length(d.indices))
        random_batch_number = popat!(d.indices, random_index)
        return (d.batches[random_batch_number], state + 1)
    end
end

function Flux.gpu(d::RNNBatches)
    return RNNBatches([gpu(d.batches[i]) for i = 1:d.length],d.indices,d.length,d.rng)  # Return a new struct with the modified field
end

Flux.@layer :expand RNNBatches trainable=(batches,)