"""
    Recurrent version of NetworkCell
    ⋅ The states V and X are updated every time Network is called as a function.
    ⋅ The states can be reset to 0 by calling the function reset.
    
    When initializing or running Network as an RNN:
    ⋅ Vseq is the sequence of voltage samples for the neurons we want to estimate (states)
    ⋅ Useq is the sequence of voltage samples for the neurons not being estimated (inputs)
        - It is a sequence of empty vectors if there are no external voltage inputs
"""
mutable struct Network{C<:NetworkCell,TV<:Tuple,TX<:Tuple}
    cell::C
    V::TV
    X::TX
    size::Tuple{Int,Int}
end

function Base.getindex(m::Network, i::Int, j::Int)
    return m.cell.ANN[i,j]
end

# Constructor with zero initial states
function Network(net::NetworkCell; nshoots=1)
    V = Tuple([zeros(Float32,1,nshoots) for i=1:net.size[1]])
    X = Tuple([Tuple([zeros(Float32,size(net.FB[i,j].state0,1),nshoots) for j=1:net.size[2]]) for i=1:net.size[1]])
    return Network(net,V,X,net.size)
end

# Constructor with warmed-up initial states
function Network(netcell::NetworkCell, d::MSData)
    return Network(netcell,d.V₀,d.X₀,net.size)
end

# Reset states to zero
function Flux.reset!(net::Network; nshoots::Int=1)
    net.V = Tuple([zeros(Float32,1,nshoots) for i=1:net.size[1]])
    net.X = Tuple([Tuple([zeros(Float32,size(net.cell.FB[i,j].state0,1),nshoots) for j=1:net.size[2]]) for i=1:net.size[1]])
end

# Reset states with initial conditions
function Flux.reset!(net::Network, data::MSData)
    net.V = map(v₀->v₀.value,data.V₀)
    net.X = map(X₀->map(x₀->x₀.value,X₀),data.X₀)
end

# Forward dynamics ("current clamp"): single step
function (net::Network)(U::Union{Nothing,Tuple},I::Tuple)
    net.V,net.X,V,X = net.cell(net.V,net.X,U,I)
    return V,X
end

# Forward dynamics with generalised teacher forcing (must have  0 ≤ γ ≤ 1): single step
function teacher(net::Network,U::Union{Nothing,Tuple},I::Tuple,V̄::Tuple)
    net.V,net.X,V,X = teacher(net.cell,net.V,net.X,U,I,V̄)
    return V,X
end

# Inverse dynamics ("voltage clamp"): single step
function inverse(net::Network,V::Tuple,U::Union{Nothing,Tuple},I::Tuple)
    ΔV,net.X,X = inverse(net.cell,V,net.X,U,I)
    return ΔV,X
end

# Forward dynamics ("current clamp"): full simulation from multiple-shooting data
function (net::Network)(data::MSData)
    # Reset initial conditions
    reset!(net,data)
    # Remaining iterations
    V̂seq = Vector{typeof(net.V)}(undef, data.shotsize)
    X̂seq = Vector{typeof(net.X)}(undef, data.shotsize)
    # Simulate the system
    for k in 1:data.shotsize
        for s in 1:data.samplingFactor
            V̂seq[k],X̂seq[k] = net(data.Useq[k],data.Iseq[k]) 
        end
    end
    return V̂seq,X̂seq
end

# Forward dynamics generalised teacher forcing (must have  0 ≤ γ ≤ 1): full simulation from multiple-shooting data
function teacher(net::Network,data::MSData)
    # Reset initial conditions
    reset!(net,data)
    # Remaining iterations
    V̂seq = Vector{typeof(net.V)}(undef, data.shotsize)
    X̂seq = Vector{typeof(net.X)}(undef, data.shotsize)
    # Simulate the system
    for k=1:data.shotsize
        for s=1:data.samplingFactor
            V̂seq[k],X̂seq[k] = teacher(net,data.Useq[k],data.Iseq[k],data.Vseq[k])
        end
    end
    return V̂seq,X̂seq
end

# Inverse dynamics ("voltage clamp"): full simulation from multiple-shooting data
function inverse(net::Network,data::MSData)
    # Reset initial conditions
    reset!(net,data)
    # Remaining iterations
    ΔV̂seq = Vector{typeof(net.V)}(undef, data.shotsize)
    X̂seq = Vector{typeof(net.X)}(undef, data.shotsize)
    for k=1:data.shotsize
        for s=1:data.samplingFactor
            ΔV̂seq[k],X̂seq[k] = inverse(net,data.Vseq[k],data.Useq[k],data.Iseq[k])
        end
    end
    return ΔV̂seq,X̂seq
end

# Forward dynamics ("current clamp"): full simulation from IO data
function (net::Network)(data::IOData)
    msData = MSData(net,data)
    V̂seq,X̂seq = net(msData)
    V̂,X̂ = shotTrajectories(V̂seq,X̂seq)
    return msData.rawdata[1].t,V̂[1],X̂[1],msData.rawdata[1].V,msData.rawdata[1].I,msData.rawdata[1].T   # There is only one shot, return it.
end

# Forward dynamics generalised teacher forcing (must have  0 ≤ γ ≤ 1): full simulation from full IO data
function teacher(net::Network,data::IOData)
    msData = MSData(net,data)
    V̂seq,X̂seq = teacher(net,msData)
    V̂,X̂ = shotTrajectories(V̂seq,X̂seq)
    return msData.rawdata[1].t,V̂[1],X̂[1],msData.rawdata[1].V,msData.rawdata[1].I,msData.rawdata[1].T    # There is only one shot, return it.
end

# Forward dynamics ("current clamp"): full simulation for constant current
function (net::Network)(v₀::F,Iapp₀::F,t::A) where {F<:AbstractFloat, A<:AbstractArray}
    V = Tuple([v₀*ones(1,length(t)) for i=1:net.size[1]])
    Iapp = Tuple([Iapp₀*ones(1,length(t)) for i=1:net.size[1]])
    ioData = IOData(V,Iapp,nothing,t,t[2]-t[1])
    return net(ioData)
end

function (net::Network)(v₀::F,Iapp₀::F,Iappₜ::K,t::A) where {F<:AbstractFloat, K<:AbstractMatrix, A<:AbstractArray}
    V = Tuple([v₀*ones(1,length(t)) for i=1:net.size[1]])
    Iapp = Tuple(Iapp₀.+Iappₜ for i=1:net.size[1])
    ioData = IOData(V,Iapp,nothing,t,t[2]-t[1])
    return net(ioData)
end

function leakCurrents(net::Network,V::Tuple)
    return [leakCurrent(net[i,j],V[j]) for i=1:net.size[1], j=1:net.size[2]]
end

function ionicCurrents(net::Network,V::Tuple,X::Tuple)
    return [ionicCurrents(net[i,j],V[i],X[i][j]) for i=1:net.size[1], j=1:net.size[2]]
end

"""
    shotTrajectories: Converts from sequence of spaced samples to (plottable) voltage and state trajectories
    Returns a tuple V[#shot][#neuron_i][#voltages] and a matrix X[#shot][#neuron_i][#neuron_j][#states]
    If there is a single shot in the data (i.e. no multiple shooting), the full data is V[1],X[1]
"""
function shotTrajectories(Vseq,Xseq)
    shotSize = length(Vseq)
    Nshots = size(Vseq[1][1],2)
    V = Vector{typeof(Vseq[1])}(undef,Nshots)
    X = Vector{typeof(Xseq[1])}(undef,Nshots)
    for t in 1:Nshots
        V[t] = ntuple(i -> reduce(hcat,[Vseq[k][i][:,t] for k in 1:shotSize]), length(Vseq[1]))
        X[t] = ntuple(i ->
                    ntuple(j -> reduce(hcat,[Xseq[k][i][j][:,t] for k = 1:shotSize]), length(Xseq[1][1])),
                length(Xseq[1]))
    end
    return V,X
end

function capacitances(net::Network)
    return Tuple(1 / weight(net.cell.Cinv[i])[1,1] for i=1:net.size[1])
end

function conductances(net::Network,V::Tuple,X::Tuple)
    return Tuple(Tuple(conductances(net.cell.ANN[i,j],V[i],X[i][j]) for j=1:net.size[2]) for i=1:net.size[1])
end

function activations(net::Network,V::Tuple,X::Tuple)
    return Tuple(Tuple(activations(net.cell.ANN[i,j],V[i],X[i][j]) for j=1:net.size[2]) for i=1:net.size[1])
end


function reversals(net::Network)
    return Tuple(Tuple(reversals(net.cell.ANN[i,j]) for j=1:net.size[2]) for i=1:net.size[1])
end

function getTimeConstants(net::Network)
    return Tuple(Tuple(getTimeConstants(net.cell.FB[i,j]) for j=1:net.size[2]) for i=1:net.size[1])
end

function IV(net::Network,Vₛ::AbstractVecOrMat,V₀::AbstractVecOrMat)
    return IV(net.cell,Float32.(Vₛ),Float32.(V₀))
end

function ssActivations(net::Network,Vₛ::AbstractVecOrMat,V₀::AbstractVecOrMat)
    return ssActivations(net.cell,Float32.(Vₛ),Float32.(V₀))
end

function findBifurcation(net::N,(i,j)::Tuple{Int,Int};tol=1e-3,Ω₀=nothing,V₀=nothing) where N<:Network
    Ω = isnothing(Ω₀) ?  [2*pi ./ getTimeConstants(net)[i,j]; 0.0] : [Ω₀;]
    V = isnothing(V₀) ?  [-25.0 ; -35.0; -45; -55.0; -65.0] : [V₀;]
    bifs = []
    for v₀ in V
        for ω₀ in Ω
            res = findBifurcation(net.cell,(i,j),v₀,ω₀)
            if res[1] == true
                add = true
                for b in bifs
                    if abs(b.v - res[2].v) < tol && abs(b.ω - res[2].ω) < tol
                        add = false
                    end
                end
                add ? push!(bifs,res[2]) : nothing
            end
        end
    end
    return bifs
end

Flux.@layer :expand Network trainable=(cell,)

function getInitTimeIndex(net::Network)    
    λ = getLargestTimescale(net.cell)
    k = log(λ,net.cell.transientPercent) # number of dt_model periods to dump
    t₀ = ceil(Int,k)
    return t₀
end

function getInitTimeIndex(τ::AbstractArray,transientPercent::AbstractFloat,dt::AbstractFloat)    
    λmax = maximum(Float32.(exp.(-dt./τ)))
    k = log(λmax,transientPercent) # number of dt_model periods to dump
    return ceil(Int,k)
end

""" 
    Custom split layer
"""
struct Split{T}
    paths::T
end
  
Split(paths...) = Split(paths)

Base.getindex(m::Split, i::Int) = m.paths[i]
Base.iterate(m::Split) = iterate(m.paths)

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

Flux.@layer Split