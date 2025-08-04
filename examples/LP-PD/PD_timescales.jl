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
data = Dict("Experiment 22/PD1"=>(id="Experiment 22/973_130_00",    numbers=[9:12;16:-1:13]),   #13:16
            "Experiment 23/PD1"=>(id="Experiment 23/973_143_00",    numbers=[1:4;9:12]),   #9:12 
            "Experiment 24/PD1"=>(id="Experiment 24/973_143_1_00",  numbers=[1:4;9:12]),   #9:12
            "Experiment 24/PD2"=>(id="Experiment 24/973_143_1_00",  numbers=[21:24;29:32]),   #29:32
            "Experiment 25/PD1"=>(id="Experiment 25/973_146_00",    numbers=[1:4;9:12]))      #9:12   

m,n = 1,1           # Number of neurons to model, number of neurons in the network
dt_data = 0.1       # [ms] , f_s = 10kHz

# Decide which recorded files will be used for training
trainfiles=3:6      # Choose subset of trial datasets for training
valfile=3           # Choose validation trial dataset
trFB=false          # No training of filter banks (redundant since we only use TF)
filterdata=false    # No extra filtering of training / validation data
propTrain = 0.35    # Proportion of data used in each training batch
propVal = 0.15      # Proportion of data used for validation

# Load and save training and validation data in appropriate format
valDataDict = Dict()
trainDataDict = Dict()
for experiment in keys(data)
    filenames = [string(datapath,data[experiment][:id],lpad(i,2,"0"),".mat") for i in data[experiment][:numbers]]
    for batch = 1:2
        # Select first or second batch of training data, and use the same validation data
        fileOpt = (propTrainInit=(batch-1)*propTrain,propTrainEnd=batch*propTrain,propVal=propVal,filt=filterdata)
        traindata_all,valdata_all = loaddata(m,n,filenames,dt_data;fileOpt...)
        trainDataDict[(experiment,batch)] = traindata_all[trainfiles]
        valDataDict[(experiment,batch)] = valdata_all[valfile]
    end
end
# @save string(modelpath,"trainData.bson") trainDataDict  # Save for plotting figures later
# @save string(modelpath,"valData.bson") valDataDict      # Save for plotting figures later

## Plot selected dataset
experiment = "Experiment 22/PD1"
batch = 2
plotSingleNeuron(trainDataDict[(experiment,batch)],vlims=:auto,plotSize=(1200,800),plotIapp=true,inds=1:50000)

####################################################
## Setting the model hyperparameters up for training
####################################################
dt_model = 0.1
ltiType =  MixedFilterCell          # Use orthogonal and diagonal realizations

τs = [0.5*1.26^i for i=0:31]
τ = [τs,τs[1:14],τs[7:28],τs[21:28]]

# Lumped current: Na + Kd + A from axon and Kd + A from soma
xIndsLumped = 32 .+ (1:14)                              #
layerUnitsLumped = (20,10,1)                 #(10,10,10,1)
layerFunLumped = (tanh,tanh,identity)      #(tanh,tanh,tanh,identity)
layerTypesLumped = (Dense,Dense,Dense)

# Mechanistic ionicCurrents: act and inact layers
layerUnits = (20,10)  
layerFun = (σ,identity)
actlayerTypes = (Positive,Positive)
inactlayerTypes = (Negative,Positive)

## For testing initial current activation ranges
# seed = 123
# traindata = trainDataDict[("Experiment 22/PD1",1)]
# net0 = Network(NetworkCell(netHP,ltiType,traindata,dt_model,trainFB=trFB,rng=MersenneTwister(seed)));
# plotIV(net0)
# plotSSActivations(net0,k=1)
# plotForcedConductances(net0,valdata)
# plotForcedCurrents(net0,valdata)
# plotForcedActivations(net0,valdata,k=1)

#######################################
## Open-loop training (teacher forcing)
#######################################
# valDict = Dict()
# trainDict = Dict()
@load string(modelpath,"valTF.bson") valDict # if file exists
# @load string(modelpath,"trainTF.bson") trainDict # if file exists

Nepochs = 250
snap_interval = round(Int,Nepochs/100)
cost_type=l2norm

# Train a model for each experiment dataset]
for sd =            [1000.0,]
for batchsize_tf =  [192*200,]
for β₁ =            [0.9,]
for β₂ =            [0.9975,]                      # [0.99,0.9925,0.995,0.9975,0.999]
for lastLayerWeight=[7.5e2,]                       # [2.5e2,5.0e2,7.5e2,1e3,2e3,5e3,1e4]
for ρ₀ =            [7.5e-5,]                      # [5e-4,1e-4,5e-5,1e-5]
for seed =          [546,564,645,654]              # 123,321,231,213,312,132,456,465,
for batch =         [1,2]                          # 1,                                  # 
for shf =           [true,]                        # true
for Δ =             [0.001,]                       # [0.001,0.00125,0.00075,0.0015,0.0005,0.00175,0.002,0.00225,0.0025]  #[0.001,0.00075,0.00125]                      #[0.000625,0.001,0.0015]     #[0.001, 0.000875, 0.00075, 0.000625, 0.0005, 0.000375, 0.00025]
for experiment = ["Experiment 22/PD1","Experiment 24/PD1","Experiment 23/PD1","Experiment 25/PD1"]
    looptime = @elapsed begin 
        # Redefine currents
        layerRegWeight = (10.,10.,lastLayerWeight)
        F_currentHP = LumpedCurrentHP(layerUnitsLumped,layerFunLumped,layerTypesLumped,xIndsLumped;
                                            vNormLims=nothing,
                                            uNormLims=nothing,
                                            maxOutput=nothing,
                                            voltageInput=true,
                                            outputBias=false,
                                            regWeight=layerRegWeight)
        
        vshift = 0.0
        regConductances = 0.0

        K = (g₀ = 10.0, E₀ = -80.0, actInds = (32+14) .+ (1:22), inactInds = (32+14+22) .+ (1:8))  # actInds = 9:28
        K_currentHP = TransientCurrentHP((K[:actInds],layerUnits,layerFun,actlayerTypes),
                                            (K[:inactInds],layerUnits,layerFun,inactlayerTypes),K[:g₀],K[:E₀]; 
                                                σ=0.0, 
                                                trainCond=false,
                                                trainNernst=false,
                                                maximalBias=false,
                                                actNormLims=(-80.0,-20.0) .+ vshift,
                                                actRegWeight=(1.0,regConductances),     # 0.0 so it doesn't regularize conductances 
                                                inactNormLims=(-80.0,-30.0) .+ vshift, 
                                                inactRegWeight=(1.0,regConductances),   # 0.0 so it doesn't regularize conductances
                                                regWeight=1.0)  
        
        Ca = (g₀ = 1.0, E₀ = 120.0, actInds = 7:20, inactInds = 21:28)
        Ca_currentHP = TransientCurrentHP((Ca[:actInds],layerUnits,layerFun,actlayerTypes),
                                            (Ca[:inactInds],layerUnits,layerFun,inactlayerTypes),Ca[:g₀],Ca[:E₀];
                                                σ=10.0, 
                                                trainCond=false,
                                                trainNernst=true,
                                                maximalBias=false,
                                                actNormLims=(-80.0,-20.0) .+ vshift, 
                                                actRegWeight=(1.0,regConductances),     # 0.0 so it doesn't regularize conductances      
                                                inactNormLims=(-80.0,-30.0) .+ vshift, 
                                                inactRegWeight=(1.0,regConductances),   # 0.0 so it doesn't regularize conductances           
                                                regWeight=1.0)
        
        H = (g₀ = 1.0, E₀ = -10.0, xInds = 25:32)
        H_currentHP = ActivationCurrentHP(H[:xInds],layerUnits,layerFun,inactlayerTypes,H[:g₀],H[:E₀]; actReadoutBias=false, 
                                            σ=10.0, 
                                            trainCond=false,
                                            trainNernst=true,
                                            maximalBias=false,
                                            actNormLims=(-80.0,-30.0) .+ vshift, 
                                            actRegWeight=(1.0,regConductances),         # 0.0 so it doesn't regularize conductances  
                                            regWeight=1.0)

        # Total ionic current
        ionicCurrentHP = (F_currentHP,K_currentHP,Ca_currentHP,H_currentHP)
        # Leak current
        leakCurrentHP = OhmicLeakHP(0.3,-50.0,
                                        trainNernst=true,
                                        regNernst=false,
                                        regWeight=1.0)
        # leakCurrentHP = LinearLeakHP(g=0.3,E=-50.0,regWeight=0.0)
        # leakCurrentHP = nothing

        # Total intrinsic current
        totalCurrentHP = TotalCurrentHP(τ,leakCurrentHP,ionicCurrentHP,ionicCurrentNames=("Lumped","K","Ca","H"))
        netHP = NetworkHP(totalCurrentHP)

        # Reset data
        traindata=trainDataDict[(experiment,batch)]
        valdata = valDataDict[(experiment,batch)]

        # Reset model
        net_tf = Network(NetworkCell(netHP,ltiType,traindata,dt_model,trainFB=trFB,rng=MersenneTwister(seed)))

        # Reset ANNData
        ANNtraindata = setup_ANNData(net_tf.cell,traindata,xpu; batchsize=batchsize_tf,shuffle=shf,partial=false,rng=MersenneTwister(123));
        ANNvaldata = setup_ANNData(net_tf.cell,valdata,xpu,shuffle=false);
        Nbatches = length(ANNtraindata)

        # Reset loss
        # ρ₀ = ρexp[experiment]
        L = Loss(cost_type,ρ₀)

        # Reset optimizer
        opt = Adam(Δ, (β₁, β₂))

        # Train
        train_tf = train_ANN(L, net_tf, ANNtraindata, ANNvaldata, opt, xpu; epochs=Nepochs, snapshots_interval = snap_interval);     
        # trainDict[(exp=experiment,batch=batch,seed=seed,cost=string(cost_type),lastLayerWeight=lastLayerWeight,std=sd,Δ=Δ,β₁=β₁,β₂=β₂,ρ₀=ρ₀,batchsize=batchsize_tf,shf=shf,epochs=Nepochs)] = train_tf[:snapshots];

        # Validate
        valOpt = (prefilter=(fc_low=1/50,fc_high=1/2,order=6),threshold=5.0,smooth=(kernelType=:gaussian,std=sd),window=0)
        val_tf = validate(train_tf[:snapshots],valdata; valOpt...);
        valDict[(exp=experiment,batch=batch,seed=seed,cost=string(cost_type),lastLayerWeight=lastLayerWeight,std=sd,Δ=Δ,β₁=β₁,β₂=β₂,ρ₀=ρ₀,batchsize=batchsize_tf,shf=shf,epochs=Nepochs)] = val_tf;
        
        # Save
        # @save string(modelpath,"trainTF.bson") trainDict
        @save string(modelpath,"valTF.bson") valDict
    end
    println("Total loop time: ",floor(Int,looptime/60),"m ",floor(Int,rem(looptime,60)),"s ")
end
end
end
end
end
end
end
end
end
end
end

##
# plotValLosses(first(values(valDict)),deltaEpochIndex=25,legend=:topright,plotBest=false,legendfontsize=10,metric=:ValLoss);
# plotValVoltages(first(values(valDict)),metric=:LastEpoch,plotPercent=(0.0,0.3),filtered=true)

# val123 = valDict[only(filter(k -> k.seed == 123, keys(valDict)))];
# plotValLosses(val123,deltaEpochIndex=25,legend=:topright,plotBest=false,legendfontsize=10,
#               metric=:ValLoss);
# plotValVoltages(val123,metric=:ValLoss,plotPercent=(0.0,0.3),filtered=true)

# ##
# val213 = valDict[only(filter(k -> k.seed == 213, keys(valDict)))];
# plotValLosses(val213,deltaEpochIndex=25,legend=:topright,plotBest=false,legendfontsize=10,
#               metric=:ValLoss);
# plotValVoltages(val213,metric=:ValLoss,plotPercent=(0.0,1.0),filtered=true)
