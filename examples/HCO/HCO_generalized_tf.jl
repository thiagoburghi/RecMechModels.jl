using Distributed                  # Distributed is using for parallel validation
rmprocs(workers())                 # Reset workers (if re-running the script)
addprocs(14,exeflags="--project")
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

# Load initial model
@load string(modelpath,"blank_model_lump.bson") net_tf
net_lump = deepcopy(net_tf)

################################################
## Fixed hyperparameters
################################################
epochs = 300
snap_interval_ms = round(Int,maximum([1,epochs/100]))  # Interval for saving snapshots (in epochs)

# Mini-batching
N_minibatches = 50                                      # Number of mini-batches   
shf = true                                               # Shuffle mini-batches?
seed = 123                                              # Seed for shuffling mini-batches

# Optimizer
η = 0.01                                                # Learning rate
β₁,β₂=0.9,0.999                                         # Moment parameters

# Validation
prefilter=(fc_low=1/500,fc_high=1/10,order=3)           # Bandpass filter to isolate spikes from slow waves
threshold=5.0                                          # Thresholds bandpassed signal
window = 5                                             # Transform peaks within window into impulse (given in number of samples)
smooth=(kernelType=:laplace,std=500.0)                  # Smooths resulting spike train

# Bundle arguments
trainOpt = (epochs=epochs,print_interval=1,snapshots_interval=snap_interval_ms)
valOpt = (prefilter=prefilter,threshold=threshold,smooth=smooth,window=window)

# Multiple shooting
s = 20
ρᵥ = 500.0
ρₓ = 500.0
ic = true
slow_ic = 1
ρ₀ = 5e-8

##########################################
## Training with variable hyperparameters
##########################################
for mechSyn = [false,]
    # Load the model dictionary for saving
    # gtfValDict = Dict()
    @load string(modelpath,"valGTF_",(mechSyn ? "mech" : "lump"),".bson") gtfValDict
    if mechSyn
        model = deepcopy(net_mech)
    else
        model = deepcopy(net_lump)
    end

    # Get data in multiple shooting format
    if N_minibatches == 1
        RNNtrainData0 = MSBatches(model,traindata;shotsize=s,
                                                  train_ic = ic,
                                                  rng=(shf ? MersenneTwister(seed) : nothing)) 
    else
        N = sum(length(traindata[i].V[1])-getInitTimeIndex(model) for i=1:length(traindata))
        miniBatchSize = N/N_minibatches
        miniBatchSize = floor(Int,miniBatchSize/192)*192 # in multiple of 192 for GPU efficiency
        spb = floor(Int,miniBatchSize/s)    
        RNNtrainData0 = MSMiniBatches(model,traindata; 
                                        shotsPerBatch=spb,
                                        shotsize=s,
                                        train_ic=ic,
                                        partial=false,  # ignore partial minibatch
                                        rng=(shf ? MersenneTwister(seed) : nothing))
    end                    

    for γ = [0.0,0.01,0.025,0.05,0.1,0.2]

        (epochs=epochs,ρ₀=ρ₀,ic=ic,γ=γ,ρᵥ=ρᵥ,ρₓ=ρₓ,s=s,β₂=β₂,window=window,shf=shf,slow_ic) in keys(gtfValDict) && continue
        # Reset everything
        RNNtrainData = deepcopy(RNNtrainData0)
        net_gtf = deepcopy(model)
        net_gtf.cell.γ[1] .= γ
        net_gtf.cell.γ[2] .= γ

        opt_gtf = Adam(η, (β₁, β₂))
        trainfun = MSLoss(net_gtf,ρ₀=ρ₀,ρᵥ=ρᵥ,ρₓ=ρₓ,teacher=(γ > 0 ? true : false))

        # Training
        train_gtf = train(net_gtf,trainfun,RNNtrainData,opt_gtf; slow_ic=slow_ic, trainOpt...);

        # Validation
        val_gtf = validate(train_gtf[:snapshots],valdata; valOpt...);

        # Save the model
        # gtfTrainDict[(epochs=epochs,ρ₀=ρ₀,ic=ic,γ=γ,ρₓ=ρₓ,s=s,β₂=β₂,window=window)] = train_gtf[:snapshots]
        gtfValDict[(epochs=epochs,ρ₀=ρ₀,ic=ic,γ=γ,ρᵥ=ρᵥ,ρₓ=ρₓ,s=s,β₂=β₂,window=window,shf=shf,slow_ic)] = val_gtf

        @save string(modelpath,"valGTF_",(mechSyn ? "mech" : "lump"),".bson") gtfValDict
    end
end