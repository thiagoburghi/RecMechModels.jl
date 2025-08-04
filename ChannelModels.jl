"""
    Total current: Implements a sum of ionicCurrents, which can be of different types.
    Some points to consider:
    ⋅ Normalization of V and X is done at the level of TotalCurrent for efficiency, since more than one ionicCurrent can
    share the same normalization layer.
    ⋅ The voltage input to the leak current (whichever type) is not normalized.
    ⋅ The voltage input to GatingCurrents is not normalized.
    ⋅ The voltage input to LumpedCurrents and InstantCurrents is normalized.
"""
struct TotalCurrent{L<:Union{Nothing,AbstractCurrent},C<:NamedTuple,R<:AbstractMatrix} 
    leakCurrent::L
    ionicCurrents::C
    ionicReadout::R
end

function Base.getindex(m::TotalCurrent, sym::Symbol)
    return getfield(m,sym)
end

function ElementwiseTanh(x::AbstractMatrix)
    return tanh.(x)
end

function TotalCurrent(hp::TotalCurrentHP,data::Vector{D},FB::AbstractLTI,t₀::Int,rng::AbstractRNG) where D<:TCData
    # Compute initial voltage and states (used to initialize currents)
    V = hcat([d.V[:,t₀:end] for d in data]...)
    X = Matrix{Float32}(undef, size(FB.state0,1), 0)
    for k=1:length(data)
        f = samplingFactor(data[k].dt,FB.dt)
        U = f > 1 ? repeat(data[k].U,inner=(1,f)) : data[k].U
        X = hcat(X,warmUp(FB,U)[:,t₀*f:f:end])
    end

    # Create the leak current model
    if isnothing(hp.leakCurrentHP)
        leakCurrent = nothing
    else
        leakCurrent = Current(hp.leakCurrentHP,V,X,data,FB,t₀,rng)
    end

    # Create the ionic current models
    ionicCurrents = (; (Symbol(hp.ionicCurrentNames[i]) => Current(h,V,X,data,FB,t₀,rng) for (i, h) in enumerate(hp.ionicCurrentHP))...)

    return TotalCurrent(leakCurrent,ionicCurrents,ones(Float32,1,length(ionicCurrents)))
end

Flux.@layer :expand TotalCurrent #trainable=(leakCurrent,ionicCurrents)
function Flux.trainable(m::TotalCurrent{Nothing,C,R}) where {C<:NamedTuple,R<:AbstractMatrix}
    return (ionicCurrents=m.ionicCurrents,)
end
function Flux.trainable(m::TotalCurrent{L,C,R}) where {L<:AbstractCurrent,C<:NamedTuple,R<:AbstractMatrix}
    return (leakCurrent=m.leakCurrent,ionicCurrents=m.ionicCurrents)
end

function (m::TotalCurrent{Nothing,C,R})(v::M,x::M) where {C<:NamedTuple,R,M<:AbstractMatrix}
    ionic = vcat(map(f->f(v,x),Tuple(m.ionicCurrents))...)
    return m.ionicReadout * ionic
end

function (m::TotalCurrent{L,C,R})(v::M,x::M) where {L<:AbstractCurrent,C<:NamedTuple,R,M<:AbstractMatrix}
    ionic = vcat(map(f->f(v,x),Tuple(m.ionicCurrents))...)
    return m.leakCurrent(v) + m.ionicReadout * ionic
end

function ionicCurrents(m::TotalCurrent,v::M,x::M) where M<:AbstractMatrix
    ionic = vcat(map(f->f(v,x),Tuple(m.ionicCurrents))...)
    return m.ionicReadout' .* ionic
end

function leakCurrent(m::TotalCurrent{Nothing,C,R},v::AbstractMatrix) where {C<:NamedTuple,R<:AbstractMatrix}
    return nothing
end

function leakCurrent(m::TotalCurrent{L,C,R},v::AbstractMatrix) where {L<:AbstractCurrent,C<:NamedTuple,R<:AbstractMatrix}
    return m.leakCurrent(v)
end

function potentials(m::TotalCurrent,v::AbstractMatrix)
    return map(current -> potentials(current,v), Tuple(m.ionicCurrents))   # reminder: no need to normalize v for potentials
end

function conductances(m::TotalCurrent,v::M,x::M) where M<:AbstractMatrix
    return map(current->conductances(current,v,x),Tuple(m.ionicCurrents))
end

function activations(m::TotalCurrent,v::M,x::M) where M<:AbstractMatrix
    return map(current->(activation(current,v,x),inactivation(current,v,x)),Tuple(m.ionicCurrents))
end

function reversals(m::TotalCurrent{Nothing,C,R}) where {C<:NamedTuple,R}
    return nothing,map(current -> reversals(current), Tuple(m.ionicCurrents))
end

function reversals(m::TotalCurrent{L,C,R}) where {L<:AbstractCurrent,C<:NamedTuple,R}
    return reversals(m.leakCurrent),map(current -> reversals(current), Tuple(m.ionicCurrents))
end

function regularizer(m::TotalCurrent{Nothing,C,R},fun::F) where {C<:NamedTuple,R,F<:Function}
    return sum([regularizer(Tuple(m.ionicCurrents)[k],fun) for k = 1:length(m.ionicCurrents)])
end

function regularizer(m::TotalCurrent{L,C,R,},fun::F) where {L<:AbstractCurrent,C<:NamedTuple,R,F<:Function}
    return sum([regularizer(m.leakCurrent,fun); [regularizer(Tuple(m.ionicCurrents)[k],fun) for k = 1:length(m.ionicCurrents)]])
end

"""
    Instantaneous current:
    Mostly used for leak, but can use for nonlinear gap junction in the future
"""
struct InstantCurrent{V<:Chain} <: AbstractCurrent
    vLayer::V
    regWeight::Float32
end

function (m::InstantCurrent)(v::M) where {M<:AbstractMatrix}
    return m.vLayer(v)
end

function Current(hp::InstantCurrentHP,V,X,data,FB,t₀,rng)
    vLayer = MLP(hp.vMlpHP,rng)

    # Adjust for desired initial parameters
    if !isnothing(hp.E) && !isnothing(hp.g)
        vLayer[end].bias .= -hp.g*hp.E
        setWeight(vLayer[end],hp.g*ones(1,length(hp.g)))
    end

    return InstantCurrent(vLayer,Float32(hp.regWeight))
end

function regularizer(m::InstantCurrent,fun::F) where F
    norm = 0
    for n = 1:length(m.vLayer)
        norm += fun(weight(m.vLayer[n]))    # will regularize ionicReadout if it exists
    end
    return m.regWeight*norm
end

function reversals(m::InstantCurrent)
    g = weight(m.vLayer[end])
    return -m.vLayer[end].bias./g
end

function potentials(m::InstantCurrent,V::AbstractMatrix)
    return nothing
end

function conductances(m::InstantCurrent,V::AbstractMatrix,X::AbstractMatrix)
    return nothing
end

function maxConductances(m::InstantCurrent)
    return weight(m.vLayer[end])
end

Flux.@layer :expand InstantCurrent

"""
    Lumped current:
    The current is the output of an MLP.
"""
struct LumpedCurrent{C<:Chain,N,R<:Union{Float32,Tuple}} <: AbstractCurrent
    mlp::C
    normLayer::N
    voltageInput::Bool
    regWeight::R
end

function Current(hp::LumpedCurrentHP,V,X,data::Vector{D},FB::AbstractLTI,t₀::Int,rng::AbstractRNG) where D<:TCData
    # Create MLP
    mlp = MLP(hp.mlpHP,rng)

    # Are we using only part of the states?
    xSelector = length(hp.xInds)==size(X,1) ? identity : Selector(hp.xInds)

    # Should we orthogonalize states?
    if hp.orthogonalize 
        xReadout = Chain(xSelector,Dense(OrthogonalTransformation(FB,inds=hp.xInds),false,identity))
    else
        xReadout = xSelector
    end

    # Is voltage an input of the mlp?
    if hp.voltageInput
        inputLayer = Parallel(vcat,identity,xReadout)
        # Threshold v if required for the normalization layer
        if isnothing(hp.vNormLims)
            Vth = [minimum(V,dims=2) maximum(V,dims=2)]
        else
            V_low = hp.vNormLims[1] == :auto ? minimum(V,dims=2) : Float32.(hp.vNormLims[1])
            V_high = hp.vNormLims[2] == :auto ? maximum(V,dims=2) : Float32.(hp.vNormLims[2])
            Vth = hcat(V_low,V_high)
        end
    else
        inputLayer = xReadout
    end
    
    # Create normalization layer
    if isnothing(hp.normFunction)
        normLayer = inputLayer
    else
        X = xReadout(X)
        if isnothing(hp.uNormLims)
            X_low = minimum(X,dims=2)
            X_high = maximum(X,dims=2)
            Xth = hcat(X_low,X_high)
            Yth = hp.voltageInput ? vcat(Vth,Xth) : Xth
        else
            dc=xReadout(DCgain(FB))
            X_low = hp.uNormLims[1] == :auto ? minimum(X,dims=2) : dc*Float32.(hp.uNormLims[1])
            X_high = hp.uNormLims[2] == :auto ? maximum(X,dims=2) : dc*Float32.(hp.uNormLims[2])
            Xth = hcat(X_low,X_high)
            Yth = hp.voltageInput ? vcat(Vth,Xth) : Xth
        end
        normLayer = Chain(inputLayer,hp.normFunction(Yth,-1,1))
    end

    # Adjust for desired initial parameters
    if !isnothing(hp.maxOutput)
        Wend = weight(mlp[end])
        setWeight(mlp[end],Float32(hp.maxOutput).*Wend./sum(Wend,dims=2))
    end
    
    return LumpedCurrent(mlp,normLayer,hp.voltageInput,Float32.(hp.regWeight))
end

function (m::LumpedCurrent)(X::AbstractMatrix)
    return m.mlp(m.normLayer(X))
end

function (m::LumpedCurrent)(V::AbstractMatrix,X::AbstractMatrix)
    return m.mlp(m.normLayer((V,X)))
end

function regularizer(m::LumpedCurrent{C,N,Float32},fun::F) where {C,N,F}
    norm = 0
    for n = 1:(length(m.mlp))
        norm += fun(weight(m.mlp[n]))
    end
    return m.regWeight*norm
end

function regularizer(m::LumpedCurrent{C,N,<:Tuple},fun::F) where {C,N,F}
    norm = 0
    for n = 1:(length(m.mlp))
        norm += m.regWeight[n]*fun(weight(m.mlp[n]))
    end
    return norm
end

function reversals(m::LumpedCurrent)
    return nothing
end

function potentials(m::LumpedCurrent,V::AbstractMatrix)
    return nothing
end

function conductances(m::LumpedCurrent,V::AbstractMatrix,X::AbstractMatrix)
    return nothing
end

function activation(m::LumpedCurrent,V::AbstractMatrix,X::AbstractMatrix)
    return nothing
end

function inactivation(m::LumpedCurrent,V::AbstractMatrix,X::AbstractMatrix)
    return nothing
end

Flux.@layer :expand LumpedCurrent trainable=(mlp,)

"""
    Gating current:
    The current mimics the Hodgkin-Huxley-type conductance-based ionicCurrents.
"""
struct GatingCurrent{AL<:Union{Nothing,LumpedCurrent},IL<:Union{Nothing,LumpedCurrent},RL<:Union{Nothing,Reversal},ML<:Union{Nothing,AbstractLayer}} <: AbstractCurrent
    actLayer::AL
    inactLayer::IL
    reversalLayer::RL
    maximalLayer::ML
    regWeight::Float32
    trainActInact::Tuple{Bool,Bool}
end

function Current(hp::GatingCurrentHP,V,X,data::Vector{D},FB::AbstractLTI,t₀::Int,rng::AbstractRNG) where D<:TCData
    # Create activation and inactivation layers
    actLayer = isnothing(hp.actHP) ? nothing : Current(hp.actHP,V,X,data::Vector{D},FB::AbstractLTI,t₀::Int,rng)
    inactLayer = isnothing(hp.inactHP) ? nothing : Current(hp.inactHP,V,X,data::Vector{D},FB::AbstractLTI,t₀::Int,rng)

    # Check the dimension of the reversal layer
    if isnothing(hp.actHP) && isnothing(hp.inactHP)
        dim = length(hp.g₀)
    elseif isnothing(hp.actHP)
        dim = hp.inactHP.mlpHP.layerUnits[end]
    else
        dim = hp.actHP.mlpHP.layerUnits[end]
    end

    # Create the reversal layer
    if isnothing(hp.E₀)
        reversalLayer = nothing
    else
        reversalLayer = Reversal(hp.E₀,dim,hp.σ,regularize=hp.regNernst,
                                                trainable=hp.trainNernst,
                                                rng=rng)
    end

    # Create the conductance layer
    if isnothing(hp.g₀)
        maximalLayer = nothing
    else
        if hp.trainCond
            maximalLayer = Positive(hp.g₀.*ones(dim)/dim,hp.maximalBias)
        else
            maximalLayer = ConstantScaling(hp.g₀/dim) 
        end
    end

    return GatingCurrent(actLayer,inactLayer,reversalLayer,maximalLayer,Float32(hp.regWeight),hp.trainActInact)
end

function (m::GatingCurrent{Nothing,Nothing,R,P})(V::M) where {R<:Reversal,P<:AbstractLayer,M<:AbstractMatrix}
    return m.maximalLayer(m.reversalLayer(V))
end

function (m::GatingCurrent{A,Nothing,R,P})(V::M,X::M) where {A<:LumpedCurrent,R<:Reversal,P<:AbstractLayer,M<:AbstractMatrix}
    return m.maximalLayer(m.actLayer(X).*m.reversalLayer(V))
end

function (m::GatingCurrent{A,Nothing,R,Nothing})(V::M,X::M) where {A<:LumpedCurrent,R<:Reversal,M<:AbstractMatrix}
    return sum(m.actLayer(X).*m.reversalLayer(V),dims=1)
end

function (m::GatingCurrent{Nothing,I,R,P})(V::M,X::M) where {I<:LumpedCurrent,R<:Reversal,P<:AbstractLayer,M<:AbstractMatrix}
    return m.maximalLayer(m.inactLayer(X).*m.reversalLayer(V))
end

function (m::GatingCurrent{A,I,R,P})(V::M,X::M) where {A<:LumpedCurrent,I<:LumpedCurrent,R<:Reversal,P<:AbstractLayer,M<:AbstractMatrix}
    return m.maximalLayer(m.actLayer(X).*m.inactLayer(X).*m.reversalLayer(V))
end

function (m::GatingCurrent{A,I,R,Nothing})(V::M,X::M) where {A<:LumpedCurrent,I<:LumpedCurrent,R<:Reversal,M<:AbstractMatrix}
    return sum(m.actLayer(X).*m.inactLayer(X).*m.reversalLayer(V),dims=1)
end

function (m::GatingCurrent{A,I,Nothing,P})(V::M,X::M) where {A<:LumpedCurrent,I<:LumpedCurrent,P<:AbstractLayer,M<:AbstractMatrix}
    return m.maximalLayer(m.actLayer(X).*m.inactLayer(X))
end

function regularizer(m::GatingCurrent{Nothing,Nothing,R,<:SignDefinite},fun::F) where {R,F}
    norm = regularizer(m.reversalLayer,fun)
    norm += fun(weight(m.maximalLayer))
    return m.regWeight*norm
end

function regularizer(m::GatingCurrent{Nothing,Nothing,R,ConstantScaling},fun::F) where {R,F}
    norm = regularizer(m.reversalLayer,fun)
    return m.regWeight*norm
end

function regularizer(m::GatingCurrent{A,Nothing,R,<:SignDefinite},fun::F) where {A<:LumpedCurrent,R,F}
    norm = regularizer(m.actLayer,fun)
    norm += regularizer(m.reversalLayer,fun)
    norm += fun(weight(m.maximalLayer))
    return m.regWeight*norm
end

function regularizer(m::GatingCurrent{A,Nothing,R,ConstantScaling},fun::F) where {A<:LumpedCurrent,R,F}
    norm = regularizer(m.actLayer,fun)
    norm += regularizer(m.reversalLayer,fun)
    return m.regWeight*norm
end

function regularizer(m::GatingCurrent{A,I,R,<:SignDefinite},fun::F) where {A<:LumpedCurrent,I<:LumpedCurrent,R,F}
    norm = regularizer(m.actLayer,fun)
    norm += regularizer(m.inactLayer,fun)
    norm += regularizer(m.reversalLayer,fun)
    norm += fun(weight(m.maximalLayer))
    return m.regWeight*norm
end

function regularizer(m::GatingCurrent{A,I,R,ConstantScaling},fun::F) where {A<:LumpedCurrent,I<:LumpedCurrent,R,F}
    norm = regularizer(m.actLayer,fun)
    norm += regularizer(m.inactLayer,fun)
    norm += regularizer(m.reversalLayer,fun)
    return m.regWeight*norm
end

function regularizer(m::GatingCurrent{A,I,Nothing,<:SignDefinite},fun::F) where {A<:LumpedCurrent,I<:LumpedCurrent,F}
    norm = regularizer(m.actLayer,fun)
    norm += regularizer(m.inactLayer,fun)
    norm += fun(weight(m.maximalLayer))
    return m.regWeight*norm
end

function regularizer(m::GatingCurrent{A,I,Nothing,ConstantScaling},fun::F) where {A<:LumpedCurrent,I<:LumpedCurrent,F}
    norm = regularizer(m.actLayer,fun)
    norm += regularizer(m.inactLayer,fun)
    return m.regWeight*norm
end

function maxConductances(m::GatingCurrent)
    return isnothing(m.maximalLayer) ? nothing : weight(m.maximalLayer)
end

function activation(m::GatingCurrent{A,Nothing,R,P},V::M,X::M) where {A<:LumpedCurrent,R,P,M<:AbstractMatrix}
    return m.actLayer(X)
end

function inactivation(m::GatingCurrent{A,Nothing,R,P},V::M,X::M) where {A<:LumpedCurrent,R,P,M<:AbstractMatrix}
    return nothing
end

function conductances(m::GatingCurrent{A,Nothing,R,Nothing},V::M,X::M) where {A<:LumpedCurrent,R,M<:AbstractMatrix}
    G = sum(m.actLayer(X),dims=1)
    return G
end

function conductances(m::GatingCurrent{A,Nothing,R,P},V::M,X::M) where {A<:LumpedCurrent,R,P,M<:AbstractMatrix}
    G = m.maximalLayer(m.actLayer(X))
    return G
end

function conductances(m::GatingCurrent{A,I,R,P},V::M,X::M) where {A<:LumpedCurrent,I<:LumpedCurrent,R,P,M<:AbstractMatrix}
    G = m.maximalLayer(m.actLayer(X).*m.inactLayer(X))
    return G
end

function conductances(m::GatingCurrent{A,I,R,Nothing},V::M,X::M) where {A<:LumpedCurrent,I<:LumpedCurrent,R,M<:AbstractMatrix}
    G = sum(m.actLayer(X).*m.inactLayer(X),dims=1)
    return G
end

function normConductances(m::GatingCurrent{A,I,R,P},V::M,X::M) where {A<:LumpedCurrent,I<:LumpedCurrent,R,P,M<:AbstractMatrix}
    G = m.actLayer(X).*m.inactLayer(X)
    return G
end

function normConductances(m::GatingCurrent{A,Nothing,R,P},V::M,X::M) where {A<:LumpedCurrent,R,P,M<:AbstractMatrix}
    G = m.actLayer(X)
    return G
end

function activation(m::GatingCurrent{A,I,R,P},V::M,X::M) where {A<:LumpedCurrent,I<:LumpedCurrent,R,P,M<:AbstractMatrix}
    return m.actLayer(X)
end

function inactivation(m::GatingCurrent{A,I,R,P},V::M,X::M) where {A<:LumpedCurrent,I<:LumpedCurrent,R,P,M<:AbstractMatrix}
    return m.inactLayer(X)
end

function potentials(m::GatingCurrent,V::M) where {M<:AbstractMatrix}
    return m.reversalLayer(V)
end

function reversals(m::GatingCurrent)
    return reversals(m.reversalLayer)
end

Flux.@layer :expand GatingCurrent

function Flux.trainable(m::GatingCurrent)
    if m.trainActInact[1] && m.trainActInact[2]
        return (actLayer=m.actLayer,inactLayer=m.inactLayer,reversalLayer=m.reversalLayer,maximalLayer=m.maximalLayer)
    elseif m.trainActInact[1]
        return (actLayer=m.actLayer,reversalLayer=m.reversalLayer,maximalLayer=m.maximalLayer)
    elseif m.trainActInact[2]
        return (inactLayer=m.inactLayer,reversalLayer=m.reversalLayer,maximalLayer=m.maximalLayer)
    else
        return (reversalLayer=m.reversalLayer,maximalLayer=m.maximalLayer)
    end
end