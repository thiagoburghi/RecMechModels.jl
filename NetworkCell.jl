# NOTE: "Cell" is the name the Flux documentation gives to the
# state-space model of a discrete-time system where states are 
# dealt with explicitly. A "Cell" can be made recurrent through 
# the wrapper "Recur", which handles the states implicity.

"""
    Network of neuronal compartments
    ⋅ ANN is a matrix containing the ANNs defining the total intrinsic current of each neuron.
    ⋅ FB is a matrix such that:
        ⋅ FB[i,i] contains the filterbank of intrinsic ionic ionicCurrents of neuron i
        ⋅ FB[i,j] contains the filterbank of synaptic ionicCurrents flowing from neuron j to neuron i
    ⋅ size is a tuple (n,m) where n ≤ m and:
        ⋅ n is the number of compartments in the model. each compartment has a membrane voltage.
        ⋅ m is the total number of membrane voltages in the data, some of which are inputs.
    ⋅ γ is a proportional feedback gain that can be used for training purposes
    ⋅ dt is the sampling period.
    ⋅ 0 < transientPercent < 1 is the % of transients we allow in ANN data and use for network initialization
    
    Terminology: 
        ⋅ MLP (Multi Layer Perceptron) refers to an unconstrained fully connected 
        artificial neural network, or parallel interconnections thereof.
        ⋅ ANN refers more generally to any type of artificial neural network.
"""
struct NetworkCell{C<:Tuple,V<:Tuple,X<:Tuple,G<:Tuple}
    Cinv::C                     # Inverse capacitances
    ANN::V                      # Matrix of TotalCurrent ANNs with leak and ionic currents
    FB::X                       # Matrix of LTI filters
    γ::G
    dt::Float32
    size::Tuple{Int,Int}
    transientPercent::Float32
    trainTeacher::Bool
end

function Base.size(net::NetworkCell, dim::Int)
    return net.size[dim]
end

Flux.@layer :expand NetworkCell
function Flux.trainable(net::NetworkCell)
    net.trainTeacher ? (return (Cinv=net.Cinv,ANN=net.ANN,FB=net.FB,γ=net.γ)) : (return (Cinv=net.Cinv,ANN=net.ANN,FB=net.FB))
end

# Constructor
function NetworkCell(annHP::Matrix{TotalCurrentHP},ltiType::Type{L}, data::Vector{D}, dt ; γ=0.0, dumpTransientPercent=0.01, trainTeacher=false, trainFB=false, rng=Random.GLOBAL_RNG) where {L<:AbstractLTI,D<:IOData}
    m,n = size(annHP)
    # Initialize inverse capacitances
    Cinv = Tuple([Nonnegative(rng,1) for i=1:m])
    # Initialize filter banks
    FB = Tuple([Tuple([ltiType(annHP[i,j].τ,dt,trainable=trainFB) for j=1:n]) for i=1:m])
    t₀ = getInitTimeIndex(FB,data,dumpTransientPercent)
    # Initialize ANNs
    ANN = Tuple([Tuple([TotalCurrent(annHP[i,j],[TCData(d,i,j) for d in data],FB[i,j],t₀,rng) for j=1:n]) for i=1:m])
    # Initialize gain
    γ = Tuple([Float32.([γ;;]) for _=1:m])
    return NetworkCell(Cinv,ANN,FB,γ,Float32(dt),(m,n),Float32(dumpTransientPercent),trainTeacher)
end

function regularizer(net::NetworkCell,fun::F) where F
    norm = sum(regularizer(net.ANN[i,j],fun) for i=1:net.size[1], j=1:net.size[2])
    # norm += sum(l2norm(weight(net.Cinv[i])) for i=1:net.size[1])
    return norm
end

# Continuous-time voltage vector field
function fv̇(net::NetworkCell,V::Tuple,X::Tuple,I::Tuple)
    V̇ = map((cinv,iapp,ann,v,x)->cinv(iapp - sum(map((f,x)->f(v,x),ann,x))),net.Cinv,I,net.ANN,V,X)
end

# Discrete-time voltage vector field
function fv₊(net::NetworkCell,V::Tuple,X::Tuple,I::Tuple) 
    V₊ = map((cinv,iapp,ann,v,x)->v+net.dt*cinv(iapp - sum(map((f,x)->f(v,x),ann,x))),net.Cinv,I,net.ANN,V,X)
end

# Internal dynamics: no external voltages
function fx₊(net::NetworkCell,V::Tuple,X::Tuple,U::Nothing)
    X₊ = map((FB,X)->map((fb,v,x)->fb(x,v)[1],FB,V,X),net.FB,X)
    return X₊
end

# Internal dynamics: with external voltages
function fx₊(net::NetworkCell,V::Tuple,X::Tuple,U::Tuple)
    VU = (V...,U...)
    X₊ = map((FB,X)->map((fb,vu,x)->fb(x,vu)[1],FB,VU,X),net.FB,X)
    return X₊
end

# Forward dynamics ("current-clamp")
function (net::NetworkCell)(V::Tuple,X::Tuple,U::Union{Nothing,Tuple},I::Tuple)
    V₊ = fv₊(net,V,X,I)
    X₊ = fx₊(net,V,X,U)
    return V₊,X₊,V,X
end

# Forward dynamics used for fixed-point iterations
function forward(net::NetworkCell,V::Tuple,X::Tuple,U::Union{Nothing,Tuple},I::Tuple,V̄::Tuple,X̄::Tuple)
    V̇ =  fv̇(net,V,X,I)
    V₊ = map((v̄,v̇)->v̄.+net.dt.*v̇,V̄,V̇)
    X₊ = fx₊(net,V,X̄,U)
    return V₊,X₊,V,X
end

# Forward dynamics with generalised teacher forcing (must have  0 ≤ γ ≤ 1)
function teacher(net::NetworkCell,V::Tuple,X::Tuple,U::Union{Nothing,Tuple},I::Tuple,V̄::Tuple)
    # If storing corrected V predictions (observer formulation)
    V̂ = map((γ,v̄,v)->γ*v̄+(1.0f0.-γ)*v,net.γ,V̄,V)             # Update step
    V₊ = fv₊(net,V̂,X,I)                                           # Correction step
    X₊ = fx₊(net,V̂,X,U)
    return V₊,X₊,V,X
end

# Inverse dynamics ("voltage clamp")
function inverse(net::NetworkCell,V::Tuple,X::Tuple,U::Union{Nothing,Tuple},I::Tuple)
    X₊ = fx₊(net,V,X,U)
    ΔV = fv̇(net,V,X,I)
    return ΔV,X₊,X
end

# IV-curves
# Vₛ is the voltage *after* the voltage step in a hypothetical voltage clamp
# V₀ is the voltage *before* the voltage step in a hypothetical voltage clamp
function IV(net::NetworkCell,Vₛ::AbstractVecOrMat,V₀::AbstractVecOrMat)
    V₀ = reshape(collect(V₀),1,length(V₀))
    Vₛ = reshape(collect(Vₛ),1,length(Vₛ))
    DC = [DCgain(net.FB[i,j]) for i=1:net.size[1], j=1:net.size[2]]
    IVleak = [leakCurrent(net.ANN[i,i],Vₛ) for i=1:net.size[1]]
    IVion = [ionicCurrents(net.ANN[i,j],Vₛ,DC[i,j]*V₀) for i=1:net.size[1], j=1:net.size[2]]
    return IVion,IVleak
end

function ssActivations(net::NetworkCell,Vₛ::AbstractVecOrMat,V₀::AbstractVecOrMat)
    V₀ = reshape(collect(V₀),1,length(V₀))
    Vₛ = reshape(collect(Vₛ),1,length(Vₛ))
    DC = [DCgain(net.FB[i,j]) for i=1:net.size[1], j=1:net.size[2]]
    return [activations(net.ANN[i,j],Vₛ,DC[i,j]*V₀) for i=1:net.size[1], j=1:net.size[2]]
end

function transferFunction(net::NetworkCell,Ω::AbstractArray)
    Hjω = [hcat([transferFunction(net.FB[i,j],Ω[k],net.dt) for k=1:length(Ω)]...) for i = 1:net.size[1], j = 1:net.size[2]]
    Hjw_abs = [abs.(Hjω[i,j]) for i=1:net.size[1], j=1:net.size[2]]
    Hjw_angle = [angle.(Hjω[i,j]) for i=1:net.size[1], j=1:net.size[2]]
    return Hjw_abs,Hjw_angle
end

function localAdmittances(net::NetworkCell,V̄::AbstractVecOrMat,Ω::AbstractVecOrMat)
    V̄ = reshape(V̄,1,length(V̄))
    X̄ = [DCgain(net.FB[i,j]).*V̄ for i=1:net.size[1], j=1:net.size[2]]
    Hjω = [[transferFunction(net.FB[i,j],Ω[k],net.dt) for i = 1:net.size[1], j = 1:net.size[2]] for k=1:length(Ω)]
    ∂ψ = [[Flux.jacobian((v̄,x̄) -> ionicCurrents(net.ANN[i,j],v̄,x̄), [V̄[k];;], reshape(X̄[i,j][:,k],length(X̄[i,j][:,k]),1)) for i=1:net.size[1], j=1:net.size[2]] for k=1:length(V̄)]
    Y_v_w = [vcat([hcat([(∂ψ[k][i,j][2] * Hjω[l][i,j] + ∂ψ[k][i,j][1]) for k=1:length(V̄)]...) for l=1:length(Ω)]...) for i=1:net.size[1], j=1:net.size[2]]
    return Y_v_w
end

function generalizedIV(net::NetworkCell,V̄::AbstractArray,ω::AbstractFloat)
    dv = V̄[2]-V̄[1]  #assumes uniform spacing
    G_v_ω = localAdmittances(net,V̄,[ω;])
    dIV_ω = hcat(real.([G_v_ω[i,1][1,1] for i=1:length(V̄)])...)
    return dIV_ω,cumsum(dIV_ω,dims=2)*dv
end

function findBifurcation(net::NetworkCell,(i,j)::Tuple{Int,Int},v₀::AbstractFloat,ω₀::AbstractFloat)
    G0 = DCgain(net.FB[i,j])
    dt = net.dt
    Hjω(ω) = transferFunction(net.FB[i,j],ω,dt)
    ∂ψ(v::AbstractMatrix,x::AbstractMatrix) = Flux.jacobian((v,x) -> net.ANN[i,j](v,x), v, x)
    function f!(F,vω)
       v,ω = vω
       grad = ∂ψ([v;;],G0*v)
       Y = exp(im*ω*dt) - 1 + weight(net.Cinv[i])[1]*(dt*sum(grad[1] + grad[2] * Hjω(ω)))
       F[1] = real(Y)
       F[2] = imag(Y)
    end

    result = nlsolve(f!, [v₀,ω₀])
    if result.f_converged == false
        return false,(v=NaN,ω=NaN,Iapp=NaN)
    else
        v,ω = result.zero
        if ω < 0 || v < -100 || v > 0
            return false,(v=NaN,ω=NaN,Iapp=NaN)
        end
        if ω < 1e-6
            ω = 0
        end
        Iapp = (net.ANN[i,j]([v;;],G0*v))[1,1]
        println("Bifurcation found: v = $v, ω = $ω, I = $Iapp")
        return true,(v=v,ω=ω,Iapp=Iapp)
    end
end

function getLargestTimescale(net::NetworkCell)
    λ = 0
    for i=1:net.size[1]
        for j=1:net.size[2]
            if λ < getLargestTimescale(net.FB[i,j])
                λ = getLargestTimescale(net.FB[i,j])
            end
        end
    end
    return λ
end

function getLargestTimescale(FB::Tuple)
    λ = 0
    for i=1:length(FB)
        for j=1:length(FB[i])
            if λ < getLargestTimescale(FB[i,j])
                λ = getLargestTimescale(FB[i,j])
            end
        end
    end
    return λ
end

# this will have to be corrected for when dt_model different than dt_data
function getInitTimeIndex(net::NetworkCell,d::IOData)    
    f = samplingFactor(d.dt,net.dt)
    λ = getLargestTimescale(net)
    k = log(λ,net.transientPercent) # number of dt_model periods to dump
    k = k/f                         # number of dt_data periods to dump
    t₀ = ceil(Int,k)
    if t₀ > length(d.t)
        println("**WARNING: Not enough data to dump all transients! Using full dataset.**")
        t₀=1
    else
        return t₀
    end
end

function getInitTimeIndex(FB::Tuple,d::IOData,dumpTransientPercent::AbstractFloat)
    f = samplingFactor(d.dt,FB[1,1].dt)
    λ = getLargestTimescale(FB)
    k = log(λ,dumpTransientPercent) # number of dt_model periods to dump
    k = k/f                         # number of dt_data periods to dump
    t₀ = ceil(Int,k)
    if t₀ > length(d.t)
        println("**WARNING: Not enough data to dump all transients! Using full dataset.**")
        t₀=1
    else
        return t₀
    end
end

function getInitTimeIndex(FB::Tuple,d::Vector{D},dumpTransientPercent::AbstractFloat) where D<:IOData
    return minimum([getInitTimeIndex(FB,d[i],dumpTransientPercent) for i=1:length(d)])
end