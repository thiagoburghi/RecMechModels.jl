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
datapath = string("examples/LP-PD/data/PD/")
modelpath = string("examples/LP-PD/results/models/PD/mechanistic/")
figurepath = string("examples/LP-PD/results/figures/")
####################################
## Load training and validation data
####################################
# Load all available data
data = Dict("Experiment 22/PD1"=>(id="Experiment 22/973_130_00",numbers=[9:12;16:-1:13]),   #13:16
            "Experiment 23/PD1"=>(id="Experiment 23/973_143_00",numbers=[1:4;9:12]),        #9:12 
            "Experiment 24/PD1"=>(id="Experiment 24/973_143_1_00",numbers=[1:4;9:12]),      #9:12
            "Experiment 24/PD2"=>(id="Experiment 24/973_143_1_00",numbers=[21:24;29:32]),   #29:32
            "Experiment 25/PD1"=>(id="Experiment 25/973_146_00",numbers=[1:4;9:12]))        #9:12   

m,n = 1,1                           # Number of neurons to model, number of neurons in the network
dt_data = 0.1                       # [ms] , f_s = 10kHz
Tc = 0.25                            # cutoff period, for filtering data

# For each dataset, use the first 35% of data for training, and the last 15% for validation
fileOpt = (propTrainEnd=0.35,propVal=0.15,filt=true,order=4,ωc=2*pi/(Tc*1e-3)) #0.25
trainfiles=3:6  # Choose subset of training data for training
valfile=3       # Choose validation dataset
trFB=false      # No training of filter banks (redundant since we only use TF)

# Load and save training and validation data in appropriate format
valDataDict = Dict()
trainDataDict = Dict()
for experiment in keys(data)
    filenames = [string(datapath,data[experiment][:id],lpad(i,2,"0"),".mat") for i in data[experiment][:numbers]]
    traindata_all,valdata_all = loaddata(m,n,filenames,dt_data;fileOpt...)
    trainDataDict[experiment] = traindata_all[trainfiles]
    valDataDict[experiment] = valdata_all[valfile]
end
@save string(modelpath,"valData.bson") valDataDict  # Save for plotting figures later

## Plot selected dataset
experiment = "Experiment 22/PD1"
plotSingleNeuron(trainDataDict[experiment],vlims=:auto,plotSize=(1200,800),plotIapp=true,inds=1:50000)

####################################################
## Setting the model hyperparameters up for training
####################################################
seed = 123
dt_model = 0.1
ltiType =  MixedFilterCell

mul=1
τs = [0.5*1.26^i for i=0:31]
τ = [τs,τs[1:14],τs[7:28],τs[21:28]]
lastLayerWeight=1e3

# Lumped current: Na + Kd + A from axon and Kd + A from soma
xIndsLumped = 32 .+ (1:14)                              #
layerUnitsLumped = (20,10,1)                 #(10,10,10,1)
layerFunLumped = (tanh,tanh,identity)      #(tanh,tanh,tanh,identity)
layerTypesLumped = (Dense,Dense,Dense)
layerRegWeight = (1.0,1.0,lastLayerWeight)
F_currentHP = LumpedCurrentHP(layerUnitsLumped,layerFunLumped,layerTypesLumped,xIndsLumped;
                                    vNormLims=nothing,
                                    uNormLims=nothing,
                                    maxOutput=nothing,
                                    voltageInput=true,
                                    outputBias=false,
                                    regWeight=layerRegWeight)

# Mechanistic ionicCurrents: act and inact layers
layerUnits = (20,10)  
layerFun = (σ,identity)
actlayerTypes = (Positive,Positive)
inactlayerTypes = (Negative,Positive)

K = (g₀ = 10.0, E₀ = -80.0, actInds = (32+14) .+ (1:22), inactInds = (32+14+22) .+ (1:8))  # actInds = 9:28
K_currentHP = TransientCurrentHP((K[:actInds],layerUnits,layerFun,actlayerTypes),
                                    (K[:inactInds],layerUnits,layerFun,inactlayerTypes),K[:g₀],K[:E₀]; 
                                        σ=0.0, 
                                        trainCond=false,
                                        trainNernst=false,
                                        maximalBias=false,
                                        actNormLims=(-80.0,-20.0),
                                        inactNormLims=(-80.0,-30.0),
                                        regWeight=1.0)

Ca = (g₀ = 1.0, E₀ = 120.0, actInds = 7:20, inactInds = 21:28)
Ca_currentHP = TransientCurrentHP((Ca[:actInds],layerUnits,layerFun,actlayerTypes),
                                    (Ca[:inactInds],layerUnits,layerFun,inactlayerTypes),Ca[:g₀],Ca[:E₀];
                                        σ=10.0, 
                                        trainCond=false,
                                        trainNernst=true,
                                        maximalBias=false,
                                        actNormLims=(-80.0,-20.0),
                                        inactNormLims=(-80.0,-30.0),
                                        regWeight=1.0)#

H = (g₀ = 1.0, E₀ = -10.0, xInds = 25:32)
H_currentHP = ActivationCurrentHP(H[:xInds],layerUnits,layerFun,inactlayerTypes,H[:g₀],H[:E₀]; actReadoutBias=false, 
                                    σ=10.0, 
                                    trainCond=false,
                                    trainNernst=true,
                                    maximalBias=false,
                                    actNormLims=(-80.0,-30.0),
                                    regWeight=1.0)

# Total ionic current
ionicCurrentHP = (F_currentHP,K_currentHP,Ca_currentHP,H_currentHP)

# Leak current
leakCurrentHP = OhmicLeakHP(1.0,-50.0,
                                trainNernst=true,
                                regNernst=false,
                                regWeight=0.0) # LinearLeakHP(;g=1.0,E=-50.0,regWeight=1.0)

# Total intrinsic current
totalCurrentHP = TotalCurrentHP(τ,leakCurrentHP,ionicCurrentHP,ionicCurrentNames=("Lumped","K","Ca","H"))
netHP = NetworkHP(totalCurrentHP)

## Some testing
# traindata = trainDataDict["Experiment 22/PD1"]
# net0 = Network(NetworkCell(netHP,ltiType,traindata,dt_model,trainFB=trFB,rng=MersenneTwister(seed)));
# plotIV(net0)
# plotSSActivations(net0,k=1)
# plotForcedConductances(net0,valdata)
# plotForcedCurrents(net0,valdata)
# plotForcedActivations(net0,valdata,k=1)

#######################################
## Open-loop training (teacher forcing)
#######################################
valDict = Dict() #@load string(modelpath,"valTF.bson") valDict # if file exists
trainDict=Dict()

Nepochs = 200
snap_interval = round(Int,Nepochs/100)
batchsize_tf,shf = 192*200,true
cost_type=l2norm
sd = 500.0
Δ  = 0.001
β₁ = 0.9
β₂ = 0.99
chunk  = 1  # Not used

# Adjust with regularization constant for different noise levels
ρexp = Dict("Experiment 22/PD1"=>1e-5,#2
            "Experiment 23/PD1"=>7.5e-5,
            "Experiment 24/PD1"=>5e-5,
            "Experiment 24/PD2"=>5e-5,
            "Experiment 25/PD1"=>2e-5)

# Train a model for each experiment dataset
for experiment in keys(data)
    # Reset data
    traindata=trainDataDict[experiment]
    valdata = valDataDict[experiment]

    # Reset model
    net_tf = Network(NetworkCell(netHP,ltiType,traindata,dt_model,trainFB=trFB,rng=MersenneTwister(seed)))

    # Reset ANNData
    ANNtraindata = setup_ANNData(net_tf.cell,traindata,xpu; batchsize=batchsize_tf,shuffle=shf,partial=false,rng=MersenneTwister(seed));
    ANNvaldata = setup_ANNData(net_tf.cell,valdata,xpu,shuffle=false);
    Nbatches = length(ANNtraindata)

    # Reset loss
    ρ₀ = ρexp[experiment]
    L = Loss(cost_type,ρ₀)

    # Reset optimizer
    opt = Adam(Δ, (β₁, β₂))

    # Train
    train_tf = train_ANN(L, net_tf, ANNtraindata, ANNvaldata, opt, xpu; epochs=Nepochs, snapshots_interval = snap_interval);     
    trainDict[(exp=experiment,cost=string(cost_type),lastLayerWeight=lastLayerWeight,chunk=chunk,Δ=Δ,β₁=β₁,β₂=β₂,ρ₀=ρ₀,batchsize=batchsize_tf,epochs=Nepochs)] = train_tf;

    # Validate
    valOpt = (prefilter=true,threshold=2.0,std=sd,fc_low=1/50,fc_high=1/2)
    val_tf = validate(train_tf[:snapshots],valdata; valOpt...);
    valDict[(exp=experiment,cost=string(cost_type),lastLayerWeight=lastLayerWeight,chunk=chunk,std=sd,Δ=Δ,β₁=β₁,β₂=β₂,ρ₀=ρ₀,batchsize=batchsize_tf,epochs=Nepochs)] = val_tf;

    # Save
    @save string(modelpath,"trainTF.bson") trainDict
    @save string(modelpath,"valTF.bson") valDict
end