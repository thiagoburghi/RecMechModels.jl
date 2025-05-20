include("./../../RecMechModels.jl")
datapath = string("examples/HCO/data/")
modelpath = string("examples/HCO/results/models/")
figurepath = string("examples/HCO/results/figures/")
voltagePlotOpts = (plotPercent=(0.0,1.0),# zoom:(0.375,0.475)
                        tlabel=0.5,
                        plotSize=(1000,350),#zoom: (600,800)
                        overlay=false,#true
                        legend=false,
                        metric=:AngSep,
                        filtered=false,
                        downsample=5)

###########################################################
## Static teacher forcing analysis
###########################################################
model_type = :lump
@load modelpath*"tf/valTF_"*string(model_type)*"_batches.bson" tfValDict
# @load modelpath*"valTF_"*string(model_type)*".bson" tfValDict

## Training plots
constKeys = (   β₂=0.999,
                ρ₀=5e-6,
                filterdata=true,
                synRegWeight=1.0,
                window=5,
                τf=0.0,
                # nbatches=50,
                shf=true)
dict_tf = filterDict(tfValDict,constKeys);
bestHP,plts = plotValLosses(dict_tf,    plotSize=(700,500),
                                        title=string("Losses by number of mini-batches: shuffling ", constKeys[:shf] ? "on" : "off",""),
                                        hpName="Mini-batches",
                                        initEpochIndex=30,
                                        deltaEpochIndex=60,
                                        metric=:AngSep,
                                        plotBest=true,
                                        epochTimes=true,
                                        overlay=false,
                                        trainingYLims=(-1.2,0.0),
                                        valYTicks=[0.0,0.5,1.0]);
savefig(string(figurepath,string("tf/tf_"*string(model_type)*"_losses_",constKeys[:shf] ? "shuffling" : "no_shuffling",".svg")))
best_models = BSON.load(string(modelpath,"best_models.bson"))
best_models[Symbol(:model,model_type)] = dict_tf[bestHP[1]][Symbol("net"*string(:AngSep))]
bson(string(modelpath,"best_models.bson"), best_models)

## Best validation metrics
best_plot = plot(plts[3],size=(500,200),
                        ylims=(0.6,0.8),
                        # yticks=[0.65,0.75,0.85],
                        title=string("Best validation metric: shuffling ", constKeys[:shf] ? "on" : "off",""),
                        bottommargin=5.0Plots.mm)
savefig(best_plot,string(figurepath,string("tf/tf_"*string(model_type)*"_best_losses_",constKeys[:shf] ? "shuffling" : "no_shuffling",".svg")))
best_plot

## Voltage traces
hp = bestHP[1]
plotValVoltages(dict_tf[hp],title=string("Best model prediction: shuffling ",constKeys[:shf] ? "on" : "off","");voltagePlotOpts...,
                plotPercent=(0.0,0.8))
savefig(string(figurepath,string("tf/tf_"*string(model_type)*"_traces_",constKeys[:shf] ? "shuffling" : "no_shuffling",".svg")))

################################################
## Multiple shooting analysis
################################################
model_type = :lump
# @load string(modelpath,"ms/valMS_lump_blank_rhofinal_500.bson") msValDict
# @load string(modelpath,"valMS_",string(model_type),".bson") msValDict

## By shotsize
constKeys = (epochs=400,
             ρ₀=5e-8,
             ic=true,
             γ=0.0,
             ρᵥ=500.0,
             ρₓ=500.0,
            # s=20,
             β₂=0.999,
             window=5,
             shf=true,
             slow_ic=1,
             )

dict_ms = filterDict(msValDict,constKeys);
bestHP,plts = plotValLosses(dict_ms,plotSize=(750,500),
                            initEpochIndex=11,
                            deltaEpochIndex=50,
                            title=string("Multiple shooting losses by shot length"),
                            metric=:AngSep,
                            overlay=false,
                            plotBest=true,
                            valYTicks=[0.0,0.5,1.0]);

savefig(string(figurepath,string("ms/ms_"*string(model_type)*"_losses.svg")))
best_models = BSON.load(string(modelpath,"best_models.bson"))
best_models[:modelms] = dict_ms[bestHP[1]][Symbol("net"*string(:AngSep))]
bson(string(modelpath,"best_models.bson"), best_models)

## MS best hyperparameters
best_plot = plot(plts[3],size=(500,200),
                        ylims=(0.5,0.9),
                        yticks=[0.65,0.75,0.85],
                        title=string("Best validation metric"),
                        bottommargin=5.0Plots.mm)
savefig(best_plot,string(figurepath,string("ms/ms_"*string(model_type)*"_best_losses.svg")))
best_plot

## MS voltage traces
hp=bestHP[1]
plotValVoltages(dict_ms[hp],title=string("Best MS model prediction: " * string(hp)),;
                    voltagePlotOpts...,
                    plotSize=(1000,600),
                    plotPercent=(0.0,0.39),
                    predictionColor=palette(:default)[1],
                    linewidth=3
                    )
savefig(string(figurepath,string("ms/ms_"*string(model_type)*"_s_",hp,"_traces.svg")))

################################################
## Generalized teacher forcing analysis
################################################
model_type = :lump
# @load string(modelpath,"valGTF_"*string(model_type)*".bson") gtfValDict
@load string(modelpath,"gtf/valGTF_lump_blank_rhofinal_500.bson") gtfValDict

## By shotsize
plotMetric=:AngSep
constKeys = (epochs=300,
             ρ₀=5e-8,
             ic=true,
            #  γ=0.0,
             ρᵥ=500.0,
             ρₓ=500.0,
             s=20, #
             β₂=0.999,
             window=5,
             slow_ic=1,
             shf=true)

gtf_dict = filterDict(gtfValDict,constKeys);
pop!(gtf_dict,0.2)
bestHP,plts = plotValLosses(gtf_dict,plotSize=(750,500),
                            initEpochIndex=11,
                            deltaEpochIndex=50,
                            title=string("Losses by generalized teacher forcing gain γ"),
                            metric=plotMetric,
                            overlay=false,
                            plotBest=true,
                            trainingYLims=(:auto,4.0),
                            valYTicks=[0.0,0.5,1.0])
savefig(string(figurepath,string("gtf/gtf_"*string(model_type)*"_losses.svg")))

# Save best model to compare synaptic prediction
hp = bestHP[1]
best_models = BSON.load(string(modelpath,"best_models.bson"))
best_gtf_model = gtf_dict[hp][Symbol("net"*string(:AngSep))]
best_models[Symbol(:modelgtf,model_type)] = best_gtf_model
bson(string(modelpath,"best_models.bson"), best_models)

##
best_plot = plot(plts[3],size=(500,200),
                        # ylims=(0.6,0.9),
                        # yticks=[0.65,0.75,0.85],
                        title=string("Best validation metric: generalized TF"),
                        bottommargin=5.0Plots.mm)
savefig(best_plot,string(figurepath,string("gtf/gtf_"*string(model_type)*"_best_losses.svg")))
best_plot

## GTF voltage traces
hp = bestHP[1]
plotValVoltages(gtf_dict[hp],title=string("Best GTF model prediction: shot length = ",hp);voltagePlotOpts...)
savefig(string(figurepath,string("gtf/gtf_"*string(model_type)*"_traces.svg")))

## Filtering figure
filenames = [datapath*"rtxi_data_Trial3.mat",
             datapath*"rtxi_data_Trial5.mat"]
m,n = 2,2                           # Number of neurons to model, number of neurons in the network
dt_data = 0.1                       # in [ms]
proptrain = 0.75                    # Proportion of data used for training
propVal = 0.25                      # Proportion of data used for validation
filterdata = true                   # Option for (non-causal) filtering of the data
Tc = 0.5                            # [ms] time constant for low-pass filter
traindata,valdata_all = loaddata(m,n,filenames,dt_data,propTrainEnd=proptrain,propVal=propVal,filt=filterdata,order=3,ωc=2*pi/(Tc*1e-3))
valdata = valdata_all[1]

## Plot GTF nonlinear filter
hp = bestHP[1]
plt_neurons = plotVoltages(gtf_dict[hp][:netAngSep],valdata,
                                neurons=[1,2],
                                plotPercent=(0.3,0.6),# zoom:(0.375,0.475)
                                tbar=0.5,
                                plotSize=(800,400),
                                vticks=[-50,-30],
                                overlay=true,
                                plotIapp=false,
                                predictionColor=palette(:default)[4],
                                linewidth=3,
                                gtf=true)
savefig(plt_neurons,string(figurepath,string("gtf/gtf_traces_γ=",hp,".svg")))
plt_neurons