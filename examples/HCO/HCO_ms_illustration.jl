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

####################################
## Train model for a few epochs
####################################
@load string(modelpath,"blank_model_lump.bson") net_tf
net_ms = deepcopy(net_tf)
s = 30
seed = 123
β₂=0.999
ρᵥ = 1.0
ρ₀ = 5e-8
ρₓ = 1.0
ic = true

# Batches and data structuring
N_batches = 50
shf=true
N = sum(length(traindata[i].V[1])-getInitTimeIndex(net_ms) for i=1:length(traindata))
miniBatchSize = N/N_batches
miniBatchSize = floor(Int,miniBatchSize/192)*192 # multiples of 192 for GPU efficiency
spb = floor(Int,miniBatchSize/s)
if N_batches == 1
    RNNtrainData = MSBatches(net_ms,traindata;shotsize=s,
                                              train_ic = ic,
                                              rng=(shf ? MersenneTwister(seed) : nothing)) 
else
    spb = floor(Int,miniBatchSize/s)
    RNNtrainData = MSMiniBatches(net_ms,traindata; 
                                    shotsPerBatch=spb,
                                    shotsize=s,
                                    train_ic=ic,
                                    partial=false,
                                    rng=(shf ? MersenneTwister(seed) : nothing))
end 

# Optimizer
opt_ms = Adam(1e-2, (0.9, β₂))
opt_schedule = ((epoch=50,η=5e-3),(epoch=75,η=2.5e-3),(epoch=100,η=1.25e-3))

trainfun = MSLoss(net_ms,ρ₀=ρ₀,ρᵥ=ρᵥ,ρₓ=ρₓ,teacher=false)

epochs = 100
snapshots_interval = round(Int,maximum([1,epochs/100]))
train_ms = train(net_ms,trainfun,RNNtrainData,opt_ms;
                                            epochs=epochs,
                                            print_interval=1,
                                            snapshots_interval=snapshots_interval,
                                            opt_schedule=opt_schedule,
                                            data_snapshots=true,
                                            slow_ic=1,
                                            );
nothing
# @save modelpath*"ms_illustration.bson" train_ms

#########################################################
## Illustrate multiple shooting
#########################################################
pltsV,pltsX = [],[]
xlims1,ylims1=nothing,nothing
for i = 1:length(train_ms[:snapshots])
    net_plot = train_ms[:snapshots][i][:model]
    data_plot = train_ms[:snapshots][i][:data]
    Ni,Nf = (400,445).+250
    pltV,pltX = multiple_shooting_plot(net_plot,data_plot,(Ni,Nf); 
                                    tUnit=:ms,
                                    neuron_index = 1,
                                    dataColor=:grey,
                                    trajColor=:red,
                                    icColor=:red,
                                    icSize=3,
                                    plotStates=true)
    if i==1
        xlims1=xlims(pltV)
        ylims1=ylims(pltV)
    end

    plot!(pltV,[xlims(pltV)[1],xlims(pltV)[1]+10],ylims(pltV)[1]*[1,1];
                                    yticks=([-20,-35],["-20mV","-35mV"]),
                                    xlims=xlims1,
                                    ylims=ylims1,
                                    xticks=false,
                                    color=:black,
                                    linewidth=3,
                                    label=false,
                                    xlabel="",
                                    bottommargin=10Plots.mm,
                                    title="Trained Epochs: "*string(train_ms[:snapshots][i][:epoch]))
    annotate!(pltV,xlims(pltV)[1]+5, ylims(pltV)[1],text("10 ms",:top,Plots.default(:fontfamily)))
    push!(pltsV,pltV)
    push!(pltsX,pltX)
end

##
for i=1:length(pltsV)
    display(pltsV[i])
    flush(stdout)
    # display(plot(pltsX[i][5],pltsX[i][30],layout=(2,1)))
    sleep(1)
end