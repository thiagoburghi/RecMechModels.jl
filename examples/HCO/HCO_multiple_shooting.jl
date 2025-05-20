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
epochs = 400
snap_interval_ms = round(Int,maximum([1,epochs/100]))  # Interval for saving snapshots (in epochs)

# Mini-batching
N_minibatches = 50                                      # Number of mini-batches   
shf = true                                              # Shuffle mini-batches?
seed = 123                                              # Seed for shuffling mini-batches

# Optimizer
η = 0.01                                               # Learning rate
β₁,β₂=0.9,0.999                                         # Moment parameters
opt_schedule = ((epoch=50,η=5e-3),(epoch=75,η=2.5e-3),(epoch=100,η=1.25e-3))

# Validation
prefilter=(fc_low=1/500,fc_high=1/10,order=3)           # Bandpass filter to isolate spikes from slow waves
threshold=5.0                                          # Thresholds bandpassed signal
window = 5                                             # Transform peaks within window into impulse (given in number of samples)
smooth=(kernelType=:laplace,std=500.0)                  # Smooths resulting spike train

# Bundle arguments
trainOpt = (epochs=epochs,print_interval=1,opt_schedule=opt_schedule,snapshots_interval=snap_interval_ms)
valOpt = (prefilter=prefilter,threshold=threshold,smooth=smooth,window=window)

# Multiple shooting
ρᵥ = 500.0
ρₓ = 500.0
ic = true

##########################################
## Training with variable hyperparameters
##########################################
for mechSyn = [false,] # true
    # Load the dict
    msValDict = Dict()
    msTrainDict = Dict()
    # @load string(modelpath,"valMS_",(mechSyn ? "mech" : "lump"),".bson") msValDict
    # @load string(modelpath,"trainMS_",(mechSyn ? "mech" : "lump"),".bson") msTrainDict
    if mechSyn
        model = deepcopy(net_mech)
    else
        model = deepcopy(net_lump)
    end

    # Shot size loop
    for γ = [0.0,]
    for s = [10,20,30,40,]
        # Organize data in MS format
        if N_minibatches == 1
            MStrainData0 = MSBatches(model,traindata; 
                                                    shotsize=s,
                                                    train_ic = ic,
                                                    rng=(shf ? MersenneTwister(seed) : nothing)) 
        else
            N = sum(length(traindata[i].V[1])-getInitTimeIndex(model) for i=1:length(traindata))
            miniBatchSize = N/N_minibatches
            miniBatchSize = floor(Int,miniBatchSize/192)*192 # multiple of 192 for GPU efficiency
            spb = floor(Int,miniBatchSize/s)
            MStrainData0 = MSMiniBatches(model,traindata; 
                                                        shotsize=s,
                                                        shotsPerBatch=spb,
                                                        train_ic=ic,
                                                        partial=false,
                                                        rng=(shf ? MersenneTwister(seed) : nothing))
        end                    
        println(length(MStrainData0))

        # Regularization loops
        for ρ₀ = [5e-8,]    # 5e-8,
        for slow_ic = [1,]
            # Skip the loop if already trained
            (epochs=epochs,ρ₀=ρ₀,ic=ic,γ=γ,ρᵥ=ρᵥ,ρₓ=ρₓ,s=s,β₂=β₂,window=window,shf=shf,slow_ic) in keys(msValDict) && continue
            
            # Reset everything
            MStrainData = deepcopy(MStrainData0)
            net_ms = deepcopy(model)
            opt_ms = Adam(η,(β₁, β₂))
            trainfun = MSLoss(net_ms,ρ₀=ρ₀,ρᵥ=ρᵥ,ρₓ=ρₓ,teacher=(γ > 0 ? true : false))
            net_ms.cell.γ[1] .= γ
            net_ms.cell.γ[2] .= γ

            # Training
            train_ms = train(net_ms,trainfun,MStrainData,opt_ms; slow_ic=slow_ic, trainOpt...);

            # Validation
            val_ms = validate(train_ms[:snapshots],valdata; valOpt...);

            # Save the model
            msTrainDict[(epochs=epochs,ρ₀=ρ₀,ic=ic,γ=γ,ρᵥ=ρᵥ,ρₓ=ρₓ,s=s,β₂=β₂,window=window,shf=shf,slow_ic)] = train_ms[:snapshots]
            msValDict[(epochs=epochs,ρ₀=ρ₀,ic=ic,γ=γ,ρᵥ=ρᵥ,ρₓ=ρₓ,s=s,β₂=β₂,window=window,shf=shf,slow_ic)] = val_ms

            @save string(modelpath,"trainMS_",(mechSyn ? "mech" : "lump"),".bson") msTrainDict
            @save string(modelpath,"valMS_",(mechSyn ? "mech" : "lump"),".bson") msValDict
        end
        end
    end
    end
end