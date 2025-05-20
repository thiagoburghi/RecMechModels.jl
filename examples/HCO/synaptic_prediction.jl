include("./../../RecMechModels.jl")
datapath = string("examples/HCO/data/")
modelpath = string("examples/HCO/results/models/")
figurepath = string("examples/HCO/results/figures/")
################################################
## Prediction of synaptic currents
################################################
m,n = 2,2                           # Number of neurons to model, number of neurons in the network
dt_data = 0.1                       # [ms] , f_s = 10kHz
proptrain = 0.75                    # Proportion of data used for training
propVal = 0.25
filterdata = true     # [ms] time constant for low-pass filter
Tc = 0.5
# Load training data from selected experiment
filenames = [datapath*"rtxi_data_Trial3.mat",
             datapath*"rtxi_data_Trial5.mat"]
_,valdata_all = loaddata(m,n,filenames,dt_data,propTrainEnd=proptrain,propVal=propVal,filt=filterdata,order=3,ωc=2*pi/(Tc*1e-3))
valdata = valdata_all[1]

# Load best models and plot one
@load string(modelpath,"best_models.bson") modellump modelmech modelms modelgtflump
model_training_type = :gtflump
@eval model = $(Symbol("model", model_training_type))
γ = 0.0
model.cell.γ[1] .= γ
model.cell.γ[2] .= γ

# Plot voltage and synaptic currents
plts=[]
for (n,m) in [(1,2),(2,1)]
    plt = plotCurrents(model,
                            valdata,
                            vticks=[-30,-50],
                            n=n,m=m,
                            plotPercent=(0.3,0.8),
                            gtf=(γ>0 ? true : false),
                            overlay=false,
                            trueData=true,
                            offsetTrue=true,
                            tbar=0.5,
                            downsample=5,
                            currentColor=palette(:default)[2])
    push!(plts,plt)
    savefig(plt,string(figurepath,string("rtxi_traces_neuron",n,".png")))
end
plt = plot(plts...,layout=(2,1),size=(400,600))
savefig(plt,figurepath*"synapse/rtxi_traces_"*string(model_training_type)*".svg")
plt