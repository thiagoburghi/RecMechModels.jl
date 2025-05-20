using Distributed
rmprocs(workers())
addprocs(7,exeflags="--project")
@everywhere begin
    using Pkg
    Pkg.activate(".")
    include("./../../RecMechModels.jl")
end
datapath = string("examples/HCO/data/")
modelpath = string("examples/HCO/results/models/")
figurepath = string("examples/HCO/results/figures/")

####################################
## Load training and validation data
####################################
# Data parameters
m,n = 2,2                           # Number of neurons to model, number of neurons in the network
dt_data = 0.1                       # in [ms]
proptrain = 0.75                    # Proportion of data used for training
propVal = 0.25                      # Proportion of data used for validation
filterdata = true                   # Option for (non-causal) filtering of the data
Tc = 0.5                            # [ms] time constant for low-pass filter

# Load training data from selected experiment
filenames = [datapath*"rtxi_data_Trial3.mat",
             datapath*"rtxi_data_Trial5.mat"]
traindata,valdata_all = loaddata(m,n,filenames,dt_data,propTrainEnd=proptrain,propVal=propVal,filt=filterdata,order=3,ωc=2*pi/(Tc*1e-3))
valdata = valdata_all[1]

#############################################
## Setting the model up
#############################################
seed = 123
dt_model = 0.1

# Time constants
τ_fast = [0.2:0.2:1.6;]
τ_slow1 = [2.0:2:16.0;]
τ_slow2 = [20.0:10.0:90.0;]
τ_ultraslow = [100.0:200.0:1500;]

# Network topology
A = [1 1; 1 1]

# Lumped intrinsic current hyperparameters
τint = [τ_fast; τ_slow1; τ_slow2; τ_ultraslow]
layerUnits_intrinsic = (20,20,20,1)                # layerUnits_intrinsic[i] = number of units in ith layer
actFun_intrinsic = (tanh,tanh,tanh,identity)       # actFun_intrinsic[i] = activation function of the ith layer

# Intrinsic current model
intrinsicHP = LumpedCurrentHP(layerUnits_intrinsic,actFun_intrinsic,1:32,voltageInput=true)
leakIntrinsicHP = LinearLeakHP()
totalIntrinsicHP = TotalCurrentHP(τint,leakIntrinsicHP,(intrinsicHP,),
                                    realization=OrthogonalFilterCell,
                                    ionicCurrentNames=("int",))

# Total training dataset length
N = sum(length(traindata[i].V[1])-getInitTimeIndex(τint,0.01,dt_model) for i=1:length(traindata))

# plotHCO(valdata_all[1],plotIapp=true,)

################################################
## Fixed hyperparameters
################################################
epochs = 600
snap_interval = round(Int,maximum([1,epochs/200]))

# Optimizer
η,β₁,β₂ = 0.001,0.9,0.999

# Validation
prefilter=(fc_low=1/500,fc_high=1/10,order=3)           # Bandpass filter to isolate spikes from slow waves
threshold=5.0                                           # Thresholds bandpassed signal
window = 5                                              # Transform peaks within window into impulse (given in number of samples)
smooth=(kernelType=:laplace,std=500.0)                  # Smooths resulting spike train
valOpt = (prefilter=prefilter,threshold=threshold,smooth=smooth,window=window)

# Cost function filtering (not used in paper)
τf = 0.0

#########################################
## Training with variable hyperparameters
#########################################
# Type of model
for mechSyn = [false,true]
    # Load the dict
    tfValDict = Dict()
    # @load string(modelpath,"valTF_",(mechSyn ? "mech" : "lump"),".bson") tfValDict

    # Regularization loop
    for ρ₀ = [5e-8,]
    for synRegWeight=[1.0,]#0.0
        # Reset the model
        if mechSyn
            # Mechanistic synapse model
            τsyn = [τ_fast;τ_slow1;τ_slow2]
            g₀ = 1.0
            E = -80.0
            synapseHP = ActivationCurrentHP(1:24,(20,1),(σ,identity),(Positive,Positive),g₀,E;
                                                trainCond=true,
                                                trainNernst=false,
                                                actNormLims=(-60.0,-40.0),
                                                actRegWeight=(synRegWeight,synRegWeight))
            totalSynapticHP = TotalCurrentHP(τsyn,nothing,(synapseHP,),
                                                ionicCurrentNames=("syn",),
                                                realization=DiagonalFilterCell)
        else
            # Lumped synapse model
            τsyn = [τ_fast;τ_slow1;τ_slow2]
            layerUnits_synaptic = (10,10,10,1)
            actFun_synaptic = (tanh,tanh,tanh,identity)  
            synapseHP = LumpedCurrentHP(layerUnits_synaptic,actFun_synaptic,1:24,
                                                voltageInput=true,
                                                # maxOutput=1.0,
                                                regWeight=synRegWeight)
            totalSynapticHP = TotalCurrentHP(τsyn,nothing,(synapseHP,),
                                                ionicCurrentNames=("syn",),
                                                realization=OrthogonalFilterCell)
        end
        netHP = NetworkHP(totalIntrinsicHP,totalSynapticHP,A)
        netcell = NetworkCell(netHP,traindata,dt_model,rng=MersenneTwister(seed));
        net_tf = Network(netcell);
        # plotSSActivations(net0,n=1,m=2,vrange=(-65.0,-40.0))
        ########################

        # Mini-batch parameters
        for shf = [true,false]
        for N_batches = [10,25,50,100,200]

            # Skip parameters already in the dict
            (ρ₀=ρ₀,β₂=β₂,shf=shf,nbatches=N_batches,filterdata=filterdata,synRegWeight=synRegWeight,window=window,τf=τf) in keys(tfValDict) && continue
            τf > 0.0 && shf==true && continue

            # Reset the data
            miniBatchSize = N/N_batches
            miniBatchSize = floor(Int,miniBatchSize/192)*192 # multiples of 192 for GPU efficiency
            ANNtraindata = setup_ANNData(net_tf.cell,traindata,xpu;batchsize=miniBatchSize,shuffle=shf,partial=false,rng=MersenneTwister(seed));
            ANNvaldata = setup_ANNData(net_tf.cell,valdata,xpu,shuffle=false);

            # Define loss function
            L = Loss(l2norm,ρ₀,1/τf,dt_model,ceil(Int,log(10)*τf/dt_model))
            opt = Adam(η, (β₁, β₂))

            # Train
            train_tf = train_ANN(L, net_tf, ANNtraindata, ANNvaldata, opt, xpu; epochs=epochs, snapshots_interval = snap_interval);       

            # Teacher forcing validation
            val_tf = validate(train_tf[:snapshots],valdata;valOpt...);

            # Save training and validation data
            tfValDict[(ρ₀=ρ₀,β₂=β₂,shf=shf,nbatches=N_batches,filterdata=filterdata,synRegWeight=synRegWeight,window=window,τf=τf)] = val_tf
            @save string(modelpath,"valTF_",(mechSyn ? "mech" : "lump"),".bson") tfValDict

        end
        end
    end
    end
end