# Load workers for validating models in parallel
using Distributed
rmprocs(workers())
addprocs(7, exeflags="--project")
# Load code into workers
@everywhere begin
    using Pkg
    Pkg.activate(".")
    include("./../../RecMechModels.jl")
end
# Change folder locations as needed
xpu = gpu
datapath = "examples/HH/data/"
modelpath = "examples/HH/results/models/"
figurepath = "examples/HH/results/figures/"
####################################
## Load training and validation data
####################################
m,n=1,1     # number of neurons to be estimated, total number of neurons in the network

# Validation data
filename1 = "HH_30.0s_σmeasurement=0.0_τ=1.0_σ=5.0_seed=123_dt=0.05_Iapp0=-5.0_Iappf=-5.0γ=0.0.mat"
# Current clamp training data
filename2 = "HH_75.0s_σmeasurement=0.0_τ=1.0_σ=1.0_seed=123_dt=0.05_Iapp0=-15.0_Iappf=0.0γ=0.0.mat"
filename3 = "HH_75.0s_σmeasurement=0.0_τ=1.0_σ=1.0_seed=456_dt=0.05_Iapp0=-15.0_Iappf=0.0γ=0.0.mat"
filename4 = "HH_75.0s_σmeasurement=0.0_τ=1.0_σ=1.0_seed=789_dt=0.05_Iapp0=-15.0_Iappf=0.0γ=0.0.mat"

# Define experiment parameters
dt = 0.05                       # [ms] , f_s = 10kHz
xpu = gpu                       # change to cpu if no gpu available

# Load data
filenames = [string(datapath,filename) for filename in [filename1,filename2,filename3,filename4]]
traindata_all,valdata_all = loaddata(m,n,filenames,dt,propTrainInit=0.0,propTrainEnd=0.9,propVal=0.1)

# Separate validation from training data
valdata = valdata_all[1]
traindata = traindata_all[2:4]

# Visualize training data
plotSingleNeuron(traindata_all[2],vlims=:auto,title="Training data: current clamp",plotIapp=true)

## Save figure
# savefig(string(figurepath,"trainingData_currentClamp.png"))

####################################
## Setting the model up for training
####################################
τ = [[0.1, 0.3, 0.5, 0.7], [1.0, 3.0, 5.0, 7.0]]           # continuous-time time constants used in filter bank

fast = 1:4
slow = 5:8
layerUnits = (20,1)
layerFun = (σ,identity)
actlayerTypes = (Positive,Positive)
inactlayerTypes = (Negative,Positive)
trainOpt = (σ=0.0, trainCond=false,trainNernst=false,maximalBias=false)

# Sodium
Na = (g₀ = 100.0, E₀ = 55.0, actInds = fast, inactInds = slow)
Na_currentHP = TransientCurrentHP((Na[:actInds],layerUnits,layerFun,actlayerTypes),
                                    (Na[:inactInds],layerUnits,layerFun,inactlayerTypes),Na[:g₀],Na[:E₀];
                                    actNormLims=(-85.0,30.0),
                                    inactNormLims=(-90.0,-40.0),
                                    actRegWeight=(1.0,0.0),
                                    inactRegWeight=(1.0,0.0),
                                    trainOpt...)
# Potassium
Kd = (g₀ = 60.0, E₀ = -77.0, xInds = slow)
Kd_currentHP = ActivationCurrentHP(Kd[:xInds],layerUnits,layerFun,actlayerTypes,Kd[:g₀],Kd[:E₀];
                                    actNormLims=(-100.0,35.0),
                                    actRegWeight=(1.0,0.0),
                                    trainOpt...)

# Ionic current
ionicCurrentHP = (Na_currentHP,Kd_currentHP)

# Leak
leakCurrentHP = OhmicLeakHP(1.0,-54.4,trainNernst=false)

# Define passive layer
ltiType = DiagonalFilterCell

# Construct neural network model
totalCurrentHP = TotalCurrentHP(τ,leakCurrentHP,ionicCurrentHP,ionicCurrentNames=("Na","K"))
netHP = NetworkHP(totalCurrentHP)

## Visualize initial IV curves and activations
# seed = 123
# net0 = Network(NetworkCell(netHP,ltiType,traindata,dt,rng=MersenneTwister(seed)));
# plotIV(net0,vrange=(-120,0))
# plotSSActivations(net0,vrange=(-120,55))
# plotForcedActivations(net0,valdata)
# plotForcedConductances(net0,valdata)

#######################################
## Open-loop training (teacher forcing)
#######################################
# Setup training data
batchsize_tf,shf = 192*200,true
# batchsize_tf,shf = Inf,false
Δ  = 0.001
β₁ = 0.9
β₂ = 0.999
ρ = 1e-6

epochs_tf = 200
snap_interval = round(Int,epochs_tf/100)

# Define loss function
valDict = Dict()
snapDict = Dict()
ltiType =  DiagonalFilterCell
for seed = [123,321,456,654,789,987]

    # Redefine training data
    traindata = traindata_all[2:4]    
    valdata = valdata_all[1]

    # Restart network 
    net_tf = Network(NetworkCell(netHP,ltiType,traindata,dt,trainFB=true,rng=MersenneTwister(seed)));
    setWeight(net_tf.cell.Cinv[1],[1.0;;])

    # Restart TF data
    traindata_tf = setup_ANNData(net_tf.cell,traindata,xpu;batchsize=batchsize_tf,shuffle=shf,rng=MersenneTwister(seed));
    valdata_tf = setup_ANNData(net_tf.cell,valdata,xpu,shuffle=false);

    # Restart optimizer
    opt = Adam(Δ, (β₁, β₂))

    # Restart loss function
    L = Loss(l2norm,ρ)

    # Main training call
    train_tf = train_ANN(L, net_tf, traindata_tf, valdata_tf, opt, xpu; epochs=epochs_tf, snapshots_interval = snap_interval);                   
    snapDict[(epochs=epochs_tf,batchsize=batchsize_tf,shuffle=shf,ρ=ρ,Δ=Δ,β₁=β₁,β₂=β₂,seed=seed,ltiType=string(ltiType))] = train_tf[:snapshots]
    # @save string(modelpath,"snapTF.bson") snapDict
    
    # Open-loop validation results
    # trainlossplot = plot(1:epochs_tf,log10.(trainloss),title="log₁₀ OL training loss function")
    # vallossplot = plot(1:epochs_tf,log10.(valloss),title="log₁₀ OL validation loss function")
    # valplot = ANNplot(net_tf.cell,ANNvaldatacpu,[1])
    # plot(trainlossplot,vallossplot,valplot[1],layout=(3,1))

    # Teacher forcing validation
    std=50.0
    threshold=-Inf
    valOpt = (prefilter=(fc_low=1/50,fc_high=1/2,order=3),threshold=threshold,smooth=(kernelType=:gaussian,std=std))
    val_tf = validate(train_tf[:snapshots],valdata; valOpt...);

    # Save
    valDict[(std=std,threshold=threshold,epochs=epochs_tf,batchsize=batchsize_tf,shuffle=shf,ρ=ρ,Δ=Δ,β₁=β₁,β₂=β₂,seed=seed,ltiType=string(ltiType))] = val_tf
    # @save string(modelpath,"valTF.bson") valDict
end

###############################################
## Validation
###############################################
@load string(modelpath,"valTF.bson") valDict

constKeys = (ρ=1e-6,Δ=0.001,β₁=0.9,β₂=0.999,epochs=200,std=50.0,threshold=-Inf,ltiType="DiagonalFilterCell",shuffle=true,batchsize=192*200)
plotDict = filterDict(valDict,constKeys);

## Plot loss trajectories
metric = :ValLoss
bestHP,plts = plotValLosses(plotDict,plotBest=false,
                        deltaEpochIndex=25,
                        hpName="Seed=",
                        title="Hodgkin-Huxley RMM teacher forcing losses",
                        metric=metric)
savefig(plts[2],string(figurepath,"TF_losses_"*string(metric)*".svg"))

## Choose hyperparameter
for (i,hp) in enumerate([123,321,456,654,789,987])
    bestVal_tf = plotDict[hp]
    net = bestVal_tf[Symbol("net"*string(metric))]

    ## Plot voltage trajectories
    pltHHVoltages=plotVoltages(net,valdata;   
                                overlay=false,
                                plotPercent=(0.7,1.0),
                                vticks=[-65,-0],
                                Iticks=[-5,5],
                                tbar=0.1,
                                IUnit="[μA]",
                                legend=false,
                                predictionColor=palette(:default)[i])
    display(pltHHVoltages)
    sleep(2)
    ## Plot conductance trajectories
    pltHHConductances=plotCurrents(net,valdata;
                                gUnit="[μS]",
                                overlay=true,
                                vticks=[-65,-0],
                                plotPercent=(0.915,0.935),
                                plotConductances=true,
                                trueData=true,
                                linewidth=2.0,
                                tbar=0.02,
                                predictionColor=palette(:default)[i])
    display(pltHHConductances)
    sleep(2)

    ## Save figures
    savefig(pltHHVoltages,string(figurepath,"HH_voltages_seed",hp,"_",string(metric),".svg"))
    savefig(pltHHConductances,string(figurepath,"HH_conductances",hp,"_",string(metric),".svg"))
end