using Flux, CUDA, Plots, Random, Distributed, SharedArrays, DistributedArrays, LaTeXStrings, DSP, LinearAlgebra, MAT, DelimitedFiles, Statistics, Peaks, NLsolve
using Flux: reset!
using BSON: @save, @load
CUDA.allowscalar(false)

l2norm(x) = sum(abs2, x)                                # For L2 regularization
l1norm(x) = sum(abs, x)                                 # For L1 regularization

default(fontfamily="Computer Modern",framestyle=:grid,linewidth=2,xguidefontsize=12,tickfontsize=12,legendfontsize=12,)
# scalefontsizes(1.3)

include("./DataTypes.jl")
include("./DataUtilities.jl")
include("./Hyperparameters.jl")
include("./ANNUtilities.jl")
include("./GOBFRealization.jl")
include("./LTIModels.jl")
include("./LayerModels.jl")
include("./ChannelModels.jl")
include("./NetworkCell.jl")
include("./Network.jl")
include("./DataTypesConstructors.jl")
include("./OpenLoopLosses.jl")
include("./ClosedLoopLosses.jl")
include("./ValidationUtilities.jl")
include("./TrainingUtilities.jl")