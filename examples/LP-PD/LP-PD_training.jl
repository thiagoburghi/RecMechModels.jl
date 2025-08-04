# Load workers for validating models in parallel
using Distributed
rmprocs(workers())
addprocs(14, exeflags="--project")
# Load code into workers
@everywhere begin
    using Pkg
    Pkg.activate(".")
    include("./../../RecMechModels.jl")
end
# Change folder locations as needed
xpu = gpu
datapath = string("examples/LP-PD/data/LP-PD/")
modelpath = string("examples/LP-PD/results/models/LP-PD/")
figurepath = string("examples/LP-PD/results/figures/")
####################################
## Visualize all data
####################################
n = 1                   # n = number of neurons to estimate
m = 2                   # m = number of membrane voltage recordings
propTrain = 0.90        # Proportion (0,1) of the full dataset used for training.
propVal = 0.1           # Proportion (0,propTrain) of the full dataset used for validation.
dt = 0.1                # [ms] , f_s = 10kHz

# Load data
filenames = map(f->string(datapath,f,".mat"),["file1","file2","file3"])
traindata_all,valdata_all = loaddata(n,m,filenames,propTrainEnd=propTrain, propVal=propVal,dt)

# Plot loaded data
plotLPPD(valdata_all)

####################################
## Set the model hyperparameters
####################################
seed=123
dt_model = 0.1
ltiType = OrthogonalFilterCell

# Define filter bank time constants
τint = vcat([0.05*1.3^i for i=0:35])
τsyn = vcat([0.05*1.3^i for i=14:29])

# Define intrinsic current hyperparameters
layerUnits_intrinsic = (10,10,10,1)                # layerUnits_intrinsic[i] = number of units in ith layer
actFun_intrinsic = (tanh,tanh,tanh,identity)       # actFun_intrinsic[i] = activation function of the ith layer

intrinsicHP = LumpedCurrentHP(layerUnits_intrinsic,actFun_intrinsic,1:36,voltageInput=true)
leakIntrinsicHP = OhmicLeakHP(0.3,-50.0,trainNernst=true,regNernst=true)
totalIntrinsicHP = TotalCurrentHP(τint,leakIntrinsicHP,intrinsicHP)

# Define synaptic current hyperparameters
layerUnits_synaptic = (10,10,1)
actFun_synaptic = (tanh,tanh,identity)

synapseHP = LumpedCurrentHP(layerUnits_synaptic,actFun_synaptic,1:16,voltageInput=true)
leakSynapseHP = nothing
totalSynapticHP = TotalCurrentHP(τsyn,leakSynapseHP,synapseHP)

# Define network hyperparameters
A = [1 1]
netHP = NetworkHP(totalIntrinsicHP,totalSynapticHP,A)

##############################################
## Set training and validation hyperparameters
##############################################
# Teacher forcing training
epochs_tf = 500
batchsize_tf = 192*200                      # Use multiple of GPU bitrate for optimal performance.
ρ₀_tf = 0.0015
Δ_tf = 0.003
β₁_tf = 0.9
β₂_tf = 0.9
loss_tf = Loss(l2norm,ρ₀_tf)

# Validation
snap_interval_tf = 5
std=200
valOpt = (prefilter=(fc_low=1/50,fc_high=1/2,order=6),threshold=2.0,smooth=(kernelType=:gaussian,std=std))

#######################################
## Teacher forcing training
#######################################
dict_tf = Dict()
for filename in ("file1","file2","file3")
    # Load data
    traindata_all,valdata_all = loaddata(n,m,string(datapath,filename,".mat"), propTrainEnd=propTrain, propVal=propVal,dt)
    valdata = valdata_all[1]

    # Initialize model
    net_tf = Network(NetworkCell(netHP,ltiType,traindata_all,dt_model,rng=MersenneTwister(seed)));
    
    # Initialize optimizer
    opt_tf = Adam(Δ_tf, (β₁_tf, β₂_tf))        # 0.95 Gradient descent optimizer. Parameters chosen by trial and error.

    # Setup teacher forcing data
    traindata_tf = setup_ANNData(net_tf.cell,traindata_all,xpu;batchsize=batchsize_tf,rng=MersenneTwister(123));
    valdata_tf = setup_ANNData(net_tf.cell,valdata,xpu);

    # Main training call
    train_tf = train_ANN(loss_tf, net_tf, traindata_tf, valdata_tf, opt_tf, xpu; epochs=epochs_tf, snapshots_interval = snap_interval_tf)                 

    # Validate
    val_tf = validate(train_tf[:snapshots],valdata; valOpt...);

    # Save
    dict_tf[(filename=filename,ρ₀=ρ₀_tf,β₁=β₁_tf,β₂=β₂_tf,epochs=epochs_tf,batchsize=batchsize_tf)] = val_tf
    @save string(modelpath,"valTF.bson") dict_tf
end

################################################
## Multiple shooting training 
################################################
# Load models trained with teacher forcing and quickly check results
@load string(modelpath,"valTF.bson") dict_tf
dict_tf = filterDict(dict_tf,(ρ₀=ρ₀_tf,β₁=β₁_tf,β₂=β₂_tf,epochs=epochs_tf,batchsize=batchsize_tf));
plotValLosses(dict_tf,title="",deltaEpochIndex=25,)

## Choose one dataset to improve with regards to teacher forcing
filename="file1"
traindata_all,valdata_all = loaddata(n,m,string(datapath,filename,".mat"), propTrainEnd=propTrain, propVal=propVal,dt)
traindata = traindata_all[1]
valdata = valdata_all[1]

## Multiple shooting hyperparameters
shotsize = 30
batchsize_ms = batchsize_tf
epochs_ms = 500
snap_interval_ms = 5
ρ₀_ms = 0.0015
ρᵥ = 1.0
ρₓ = 0.1
Δ_ms = 0.003
β₁_ms = 0.9
β₂_ms = 0.9

## Multiple shooting training
# Reset model 
net_ms = deepcopy(dict_tf[filename])[:netValLoss]

# Reset train function (requires network for state regularization weights)
trainfun_ms = MSLoss(net_ms,ρ₀=ρ₀_ms,ρᵥ=ρᵥ,ρₓ=ρₓ)

# Reset data and initial conditions
msData = MSData(net_ms,traindata,train_ic=true,shotsize=shotsize)
if batchsize_ms == Inf 
    traindata_ms = msData
else
    msBatchesData = MSMiniBatches(msData,shotsPerBatch=round(Int,batchsize_ms/shotsize),rng=MersenneTwister(123));
    traindata_ms = msBatchesData
end

# Reset optimizer
opt_ms = Adam(Δ_ms, (β₁_ms, β₂_ms))

# Train
train_ms = train(net_ms,trainfun_ms,traindata_ms,opt_ms; epochs=epochs_ms, snapshots_interval=snap_interval_ms,print_interval=1);
# plot(1:length(train_ms[:loss]),log10.(train_ms[:loss]),title="Multiple-shooting training loss")

## Multiple shooting validation
val_ms = validate(train_ms[:snapshots],valdata; valOpt...);

# Save results
dict_ms = Dict() #@load string(modelpath,"valMS.bson") dict_ms  # if file already exists
dict_ms[(filename=filename,ρ₀=ρ₀_ms,ρₓ=ρₓ,Δ=Δ_ms,β₁=β₁_ms,β₂=β₂_ms,epochs=epochs_ms,batchsize=batchsize_ms,std=std,shotsize=shotsize)] = val_ms
@save string(modelpath,"valMS.bson") dict_ms

## Plot loss functions
plotValLosses(val_ms,metric=:ValLoss,title="Multiple-shooting validation")

#########################################################
## Illustration of multiple shooting intervals
#########################################################
RNNplotdata = first(train_ms[:data])
Ni,Nf = (50,90)                                 # Batch numbers to plot
plt1,pltX1 = multiple_shooting_plot(train_ms[:net],RNNplotdata,(Ni,Nf))
plt1