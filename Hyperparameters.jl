"""
    Types containing hyperparameters of an MLP. 
    ⋅ layerUnits = number of units in ith layer the ANN branch
    ⋅ layerFun = activation function used in ith layer of the ANN branch
    ⋅ xInds = indices of filter bank states used as inputs to the ANN branch
    ⋅ layertype = type of layer used in the ANN branch
"""
struct MlpHP <: AbstractCurrentHP
    nInputs::Int
    layerUnits::Tuple{Vararg{Int}}
    layerFun::Tuple{Vararg{Function}}
    layerTypes::Tuple
    readoutBias::Bool
end

struct InstantCurrentHP <: AbstractCurrentHP
    vMlpHP::MlpHP
    g::Union{Nothing,AbstractFloat}
    E::Union{Nothing,AbstractFloat}
    regWeight::AbstractFloat
end

function LinearLeakHP(;E=nothing,g=nothing,regWeight=1.0)
    return InstantCurrentHP(MlpHP(1,(1,),(identity,),(Positive,),true),g,E,regWeight)
end

function NonlinearLeakHP(vUnits,vActFun,layerTypes;regWeight=1.0)
    vUnits[end] != 1 ? error("Number of units in the last layer of the leak MLP must be 1.") : nothing
    return InstantCurrentHP(MlpHP(1,vUnits,vActFun,layerTypes,true),nothing,nothing,regWeight)
end

###########################################
## NEW VERSION OF HYPERPARAMETERS 
###########################################
struct TotalCurrentHP <: AbstractCurrentHP
    τ::AbstractVector
    leakCurrentHP::Union{Nothing,AbstractCurrentHP}
    ionicCurrentHP::Tuple{Vararg{AbstractCurrentHP}}
    ionicCurrentNames::Tuple{Vararg{String}}
end

# Full constructor using keyword arguments 
function TotalCurrentHP(τ,leakCurrentHP,ionicCurrentHP::Tuple; ionicCurrentNames::Tuple)
    if length(ionicCurrentHP) != length(ionicCurrentNames)
        error("Number of ionicCurrentHP must equal number of ionicCurrentNames")
    end
    return TotalCurrentHP(τ,leakCurrentHP,ionicCurrentHP,ionicCurrentNames)
end

# Simple constructor for single ionic current and no activation function priors
function TotalCurrentHP(τ,leakCurrentHP,ionicCurrentHP::AbstractCurrentHP)
    ionicCurrentHP = (ionicCurrentHP,)
    ionicCurrentNames = ("Intrinsic current",)
    return TotalCurrentHP(τ,leakCurrentHP,ionicCurrentHP,ionicCurrentNames)
end

struct LumpedCurrentHP <: AbstractCurrentHP
    xInds::AbstractVector
    mlpHP::MlpHP
    voltageInput::Bool
    normFunction::Union{Nothing,Function}
    vNormLims::Union{Nothing,Tuple}
    uNormLims::Union{Nothing,Tuple}
    maxOutput::Union{Nothing,AbstractFloat}
    orthogonalize::Bool
    regWeight::Union{AbstractFloat,Tuple}
end

function LumpedCurrentHP(layerUnits,layerFun,xInds; voltageInput=true, normFunction=MinMaxNorm, vNormLims=nothing, uNormLims=nothing, maxOutput=nothing, outputBias=false, orthogonalize=false, regWeight=1.0)
    nInputs = voltageInput ? 1 + length(xInds) : length(xInds)
    mlpHP = MlpHP(nInputs,layerUnits,layerFun,Tuple(fill(Dense,length(layerUnits))),outputBias)
    return LumpedCurrentHP(xInds,mlpHP,voltageInput,normFunction,vNormLims,uNormLims,maxOutput,orthogonalize,regWeight)
end

function LumpedCurrentHP(layerUnits,layerFun,layerTypes,xInds; voltageInput=true, normFunction=MinMaxNorm, vNormLims=nothing, uNormLims=nothing, maxOutput=nothing, outputBias=false, orthogonalize=false, regWeight=1.0)
    nInputs = voltageInput ? 1 + length(xInds) : length(xInds)
    mlpHP = MlpHP(nInputs,layerUnits,layerFun,layerTypes,outputBias)
    return LumpedCurrentHP(xInds,mlpHP,voltageInput,normFunction,vNormLims,uNormLims,maxOutput,orthogonalize,regWeight)
end

struct GatingCurrentHP <: AbstractCurrentHP
    actHP::Union{Nothing,LumpedCurrentHP}
    inactHP::Union{Nothing,LumpedCurrentHP}
    g₀
    E₀
    σ
    trainCond::Bool
    trainNernst::Bool
    regNernst::Bool
    maximalBias::Bool
    regWeight::AbstractFloat
    trainActInact::Tuple{Bool,Bool}
    function GatingCurrentHP(actHP::AH,inactHP::IH,args...) where {AH,IH}
        if AH<:LumpedCurrentHP && IH<:LumpedCurrentHP
            if actHP.mlpHP.layerUnits[end] != inactHP.mlpHP.layerUnits[end]
                error("Number of units in the last layer of the activation and inactivation MLPs must be the same.")
            end
        end
        return new(actHP,inactHP,args...)
    end
end

function OhmicLeakHP(g₀,E₀; σ=nothing,trainCond=true,trainNernst=true,regNernst=false,maximalBias=false,regWeight=1.0)
    return GatingCurrentHP(nothing,nothing,g₀,E₀,σ,trainCond,trainNernst,regNernst,maximalBias,regWeight,(false,false))
end

function ActivationCurrentHP(xInds,xUnits,xLayerFun,xLayerTypes,g₀,E₀; σ=nothing,
                                                                        trainCond=true,
                                                                        trainNernst=true,
                                                                        regNernst=true,
                                                                        maximalBias=false,
                                                                        actReadoutBias=false,
                                                                        actNormFunction=MinMaxNorm,
                                                                        actNormLims=nothing,
                                                                        actRegWeight=1.0,
                                                                        actOrthogonalize=false,
                                                                        regWeight=1.0)
    actMlpHP = MlpHP(length(xInds),xUnits,xLayerFun,xLayerTypes,actReadoutBias)
    actHP = LumpedCurrentHP(xInds,actMlpHP,false,actNormFunction,nothing,actNormLims,1.0,actOrthogonalize,actRegWeight) 

    return GatingCurrentHP(actHP,nothing,g₀,E₀,σ,trainCond,trainNernst,regNernst,maximalBias,regWeight,(true,false))
end

function InactivationCurrentHP(xInds,xUnits,xLayerFun,xLayerTypes,g₀,E₀; σ=nothing,
                                                                        trainCond=true,
                                                                        trainNernst=true,
                                                                        regNernst=true,
                                                                        maximalBias=false,
                                                                        inactReadoutBias=false,
                                                                        inactNormFunction=MinMaxNorm,
                                                                        inactNormLims=nothing,
                                                                        inactRegWeight=1.0,
                                                                        inactOrthogonalize=false,
                                                                        regWeight=1.0)
    inactMlpHP = MlpHP(length(xInds),xUnits,xLayerFun,xLayerTypes,inactReadoutBias)
    inactHP = LumpedCurrentHP(xInds,inactMlpHP,false,inactNormFunction,nothing,inactNormLims,1.0,inactOrthogonalize,inactRegWeight)

    return GatingCurrentHP(nothing,inactHP,g₀,E₀,σ,trainCond,trainNernst,regNernst,maximalBias,regWeight,(false,true))
end

function TransientCurrentHP((actInds,actUnits,actLayerFun,actLayerTypes),(inactInds,inactUnits,inactLayerFun,inactLayerTypes),g₀,E₀; 
                                                                        σ=nothing,
                                                                        trainCond=true,
                                                                        trainNernst=true,
                                                                        regNernst=true,
                                                                        maximalBias=false,
                                                                        actReadoutBias=false,
                                                                        inactReadoutBias=false,
                                                                        actNormFunction=MinMaxNorm,
                                                                        actNormLims=nothing,
                                                                        inactNormFunction=MinMaxNorm,
                                                                        inactNormLims=nothing,
                                                                        actRegWeight=1.0,
                                                                        inactRegWeight=1.0,
                                                                        actOrthogonalize=false,
                                                                        inactOrthogonalize=false,
                                                                        trainActInact=(true,true),
                                                                        regWeight=1.0)
    actMlpHP = MlpHP(length(actInds),actUnits,actLayerFun,actLayerTypes,actReadoutBias)
    actHP = LumpedCurrentHP(actInds,actMlpHP,false,actNormFunction,nothing,actNormLims,1.0,actOrthogonalize,actRegWeight)

    inactMlpHP = MlpHP(length(inactInds),inactUnits,inactLayerFun,inactLayerTypes,inactReadoutBias)
    inactHP = LumpedCurrentHP(inactInds,inactMlpHP,false,inactNormFunction,nothing,inactNormLims,1.0,inactOrthogonalize,inactRegWeight)

    return GatingCurrentHP(actHP,inactHP,g₀,E₀,σ,trainCond,trainNernst,regNernst,maximalBias,regWeight,trainActInact)
end

"""
    Type containing hyperparameters of the network. 
    ⋅ m is the number of neurons in the network whose model must be estimated.
    ⋅ n is the total number of neurons in the network.
    Convention: neurons whose model must be estimated are indexed by 1:m,
                neurons whose model are not estimated are index by m+1:n. 
"""
# Basic constructor for a single neuron without synapses 
function NetworkHP(totalCurrentHP::TotalCurrentHP)
    return [totalCurrentHP;;]
end

# Basic constructor with identical passive models, channel models and synapse models.
# A is an adjacency matrix such that:
#   ⋅ m = Size(A,1)
#   ⋅ n = Size(A,2)
#   A[i,i] = 1 if neuron i contains ion channels (otherwise, just an RC circuit)
#   A[i,j] = 1 if there are synapses from neuron j to neuron i
function NetworkHP(ionHP::TotalCurrentHP,synHP::TotalCurrentHP,A::AbstractMatrix)
    annHP = Matrix{TotalCurrentHP}(undef,size(A))
    for i = 1:size(A,1)
        if A[i,i] == 1
            annHP[i,i] = ionHP
        end
        for j = 1:size(A,2)
            if A[i,j] == 1 && i!=j
                annHP[i,j] = synHP                
            end
        end
    end
    return annHP
end