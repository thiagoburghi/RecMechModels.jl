## Load packages
include("./../../RecMechModels.jl")
include("./PD_plot_options.jl")
datapath = string("examples/LP-PD/data/PD/")
modelpath = string("examples/LP-PD/results/models/PD/mechanistic/")
figurepath = string("examples/LP-PD/results/figures/")

# Load training and validation voltage traces
@load string(modelpath,"valData.bson") valDataDict
@load string(modelpath,"trainData.bson") trainDataDict

####################################
## Paper Figure 4 C + SI Appendix 
####################################
@load string(modelpath,"valTF_nofilter_7.5e-5_9975.bson") valDict

## Recover validation results for selected hyperparameters and plot losses
# hp1=:seed
# hp2=:batch

hp1=  :seed  # :ρ₀ # 
hp2=  :batch # :lastLayerWeight # 

plotMetric = :ValLoss

expList = ["Experiment 22/PD1","Experiment 25/PD1","Experiment 23/PD1","Experiment 24/PD1"] #
for experiment = expList
    # Fixed hyperparameters
    hpDict = Dict(  :exp=>experiment,
                    :cost=>string(l2norm),
                    :lastLayerWeight=>7.5e2,    #1e3,#7.5e2,#
                    :std=>1000.,                # 750., 1000.0,
                    :β₁=>0.9,
                    :β₂=>0.9975,                # 0.99, 0.9975
                    :ρ₀=>7.5e-5,                  #7.5e-5,
                    :Δ=>0.001, #0.001
                    :batchsize=>192*200,
                    :shf=>true,
                    :epochs=>250, #200
                    :seed=>123,
                    :batch=>1,
                    # :regConductances=>0.0,  # 1.0
                )
    # Get trained models with different hp1 and hp2 values
    # Reduce velue removes the worst models
    dict = reduceDict(valDict,hpDict,hp1,hp2,metric=plotMetric,optMetric=0.0,reduce=6)
    
    # Plot losses and recover ranking of best hyperparameters
    bestHP,plts = plotValLosses(dict,
                                metric=plotMetric,
                                plotSize=(600,600),
                                title="Training losses: preparation "*string(expDict[experiment][:number]),
                                hull=false,
                                deltaEpochIndex=25,
                                legend=:topright,
                                plotBest=false,
                                opacityAfterFirst=0.7,
                                legendfontsize=10);

    expDict[experiment][:lossPlot] = current()
    expDict[experiment][:bestHP] = bestHP
    expDict[experiment][:valDict] = dict
end

# Plot losses for all experiments
plts = [plot(expDict[experiment][:lossPlot]) for experiment in expList]
pltLosses=plot(plts...,size=(1600,1000))
savefig(pltLosses,string(figurepath,"Losses_preps_",expDict[expList[1]][:number],"-",expDict[expList[end]][:number],".pdf"))
pltLosses

## Check results
experiment = "Experiment 22/PD1"
plotPct=(0.0,0.25)
pltsV = []
for i=1:min(length(expDict[experiment][:bestHP]),4)
    # Check voltages for each batch
    plt=plotValVoltages(expDict[experiment][:valDict][expDict[experiment][:bestHP][i]],
                        metric=plotMetric,
                        title=string(expDict[experiment][:bestHP][i]),
                        plotPercent=plotPct,
                        overlay=false,
                        filtered=true)
    push!(pltsV,plt)
end
pltV=plot(pltsV...,size=(1600,800),
    layout=(2,ceil(Int,min(length(expDict[experiment][:bestHP]),4)/2)))

#############
## IV curves
#############
IVplts = []
expGroups = [["Experiment 22/PD1","Experiment 25/PD1"],["Experiment 23/PD1","Experiment 24/PD1"]]
for expGroup in expGroups
    plt_tot,plt_leak,plt_ion = plot(),plot(),[plot() for _ in 1:4]
    local pltIV
    for experiment in expGroup
        plotIV!(expDict[experiment][:valDict],(plt_tot,plt_leak,plt_ion),
                envelope=true,
                vrange=(-65,-15),
                metric=plotMetric,
                vticks=[-60,-45,-30,-15],
                color=expDict[experiment][:color],
                hyParsName="Prep "*string(expDict[experiment][:number]),
        )
        pltIV=plot(plt_ion...,layout=(1,4),
                    size=(2.25*500,2.25*200),
                    bottommargin=5.0Plots.mm,
                    title="",   # fig 4c
                    )
    end
    push!(IVplts,pltIV)
    savefig(pltIV,string(figurepath,"IVs_preps_",expDict[expGroup[1]][:number],"-",expDict[expGroup[end]][:number],".svg"))  
end
allIV=plot(IVplts...,size=(2.25*500,2.25*400),layout=(2,1))
savefig(allIV,string(figurepath,"IVs_preps_",expDict[expGroups[1][1]][:number],"-",expDict[expGroups[2][end]][:number],".svg"))  
allIV

######################
## Individual currents
######################
Iplts = []
for experiment in expList[1:4]
    # Choose some model to plot
    ranking = 2
    hp    = expDict[experiment][:bestHP][ranking]
    model = expDict[experiment][:valDict][hp][Symbol("net"*string(plotMetric))]
    batch = expDict[experiment][:bestHP][ranking][2]

    condPlot=plotConductances(model,valDataDict[(experiment,batch)],
                        # plotPercent=(0.78,0.9),       # zoomed in fig 4c
                        # plotSize=(400,800),           # zoomed in fig 4c
                        # vlims=(-65,-15),              # zoomed in fig 4c
                        plotPercent=(0.3,0.6),      # si appendix
                        plotSize=(600,800),         # si appendix
                        tbar=0.5,
                        tickfontsize=12,
                        xlabelfontsize=10,
                        ylabelfontsize=14,
                        vticks=[-20,-40],
                        tlabel="t[s]",
                        predictionColor=expDict[experiment][:color],
                        overlay=false,
                        vtitle="Preparation "*string(expDict[experiment][:number]),
                        downsample=5
                        )
    # condPlot=plot(condPlot,left_margin=8.0Plots.mm) # zoomed in fig 4c
    push!(Iplts,condPlot)
    savefig(condPlot,string(figurepath,"conductances"*replace(experiment, "/" => "-")*".svg"))
end
plt_all = plot(Iplts...,layout=(2,2),size=(2*600,2*800),leftmargin=10.0Plots.mm)
savefig(plt_all,string(figurepath,"Conductances_preps_",expDict[expList[1]][:number],"-",expDict[expList[end]][:number],".pdf"))
plt_all

####################################
## Paper Figure 4 E (current removal)
####################################
ranking = 2
experiment = "Experiment 25/PD1"
hp    = expDict[experiment][:bestHP][ranking]
model = expDict[experiment][:valDict][hp][Symbol("net"*string(plotMetric))]
batch = expDict[experiment][:bestHP][ranking][2]
valdata = valDataDict[(experiment,batch)]

dt_data=0.1
axisOpts = (vlims=(-52.5,-10),
            plotPercent=(0.8,0.925),
            vticks=([-60,-40,-20],[L"-60",L"-40",L"-20"]),
            tbar=0.5,
            linewidth=3,
            predictionColor=expDict[experiment][:color],
            tlabelfontsize=16)
labelOpts = (legend=false,framestyle=:grid,ytickfontsize=16,xtickfontsize=16,margins=5.0Plots.mm)

# A current ramp to catch oscillations
actvaldata=deepcopy(valdata)
actvaldata.V[1][1,:] = range(-80.0, -20.0; length=length(valdata.V[1]))
actvaldata.I[1][1,:] = 5*actvaldata.I[1][1,:]
actvaldata.I[1][1,:] += range(-30.0, 30.0; length=length(valdata.I[1]))

# Full model
Ioffset_full = 0.0
current_gain = 0.0
net_full = deepcopy(model)
I_full = [current_gain*valdata.I[i] .+ ones(1,length(valdata.I[i]))*Ioffset_full for i=1:length(valdata.I)]
valdata_full = IOData(valdata.V,Tuple(I_full),valdata.T,valdata.t,Float32(dt_data))

# Plot
plt=getVoltagePlots(net_full,valdata_full;axisOpts...)
plt_full = plot(plt[1][1][:pltV̂];ylabelfontsize=16,labelOpts...)
savefig(plt_full,string(figurepath,"PD-full_removal.svg"))
plt_full

#######################################################
## Remove fast ionicCurrents and find bifurcations
#######################################################
net_su = deepcopy(model)
net_su[1,1].ionicReadout[:] = [0.0,1.0,1.0,1.0]
bifs_su = findBifurcation(net_su,(1,1),Ω₀=[0.0,0.1,0.01,0.001],V₀=[-70,-65.0,-60.0,-55.0,-50,-45.,-40.,-35.,-30.])

##
# plotVoltages(net_su,actvaldata,
#                 legend=false,
#                 tticks=false,
#                 plotPercent=(0.0,1.0))

## Simulate slow wave
# From bifurcation
bif_number = 2
v₀ = bifs_su[bif_number][:v]
Iapp_su = bifs_su[bif_number][:Iapp]
# From eyeballed values
# v₀ = -60.0
# Iapp_su = -14.0
# t_su,V_su₋ = net_su(v₀,Iapp₀,valdata.I[1],valdata.t)

# Probe around bifurcations
δ=0.1
t_su,V_su₋ = net_su(v₀,Iapp_su-δ,valdata.t)
t_su,V_su₊ = net_su(v₀,Iapp_su+δ,valdata.t)
plt_su=plot(t_su/1000,V_su₋[1][:],label="Iapp-0.1")
plt_su=plot!(t_su/1000,V_su₊[1][:],label="Iapp+0.1")

## Plot for paper
current_gain = 0.0
Ioffset_su = Iapp_su+2*δ
I_su = [current_gain*valdata.I[i] .+ ones(1,length(valdata.I[i]))*Ioffset_su for i=1:length(valdata.I)]
valdata_su = IOData(valdata.V,Tuple(I_su),valdata.T,valdata.t,Float32(dt_data))

# Plot
plt=getVoltagePlots(net_su,valdata_su;axisOpts...)
plt_su_paper = plot(plt[1][1][:pltV̂];ylabel="",labelOpts...)
savefig(plt_su_paper,string(figurepath,"PD-slow_removal.svg"))
plt_su_paper

#######################################################
## Remove slow ionicCurrents and find bifurcations
#######################################################
net_fs = deepcopy(model)
net_fs[1,1].ionicReadout[:] = [1.0,0.0,0.0,0.0]
bifs_fs = findBifurcation(net_fs,(1,1),Ω₀=[0.0,0.1,0.01,0.001],V₀=[-70,-65.0,-60.0,-55.0,-50,-45.,-40.0,-35.0,-30.0])

##
# plotVoltages(net_fs,actvaldata,
#                 legend=false,
#                 plotPercent=(0.0,1.0))
## Simulate slow wave
# From bifurcation
bif_number = 1
v₀ = bifs_fs[bif_number][:v]
Iapp_fs = bifs_fs[bif_number][:Iapp]
# From eyeballed values
# v₀ = -48.
# Iapp_fs = 11.9

# t_fs,V_fs₋ = net_fs(v₀,Iapp₀,valdata.I[1],valdata.t)
# Probe around bifurcations
δ=0.1
t_fs,V_fs₋ = net_fs(v₀,Iapp_fs-δ,valdata.t)
t_fs,V_fs₋ = net_fs(v₀,Iapp_fs+δ,valdata.t)
plt_fs=plot(t_fs/1000,V_fs₋[1][:],label="Iapp-0.1")
plt_fs=plot!(t_fs/1000,V_fs₋[1][:],label="Iapp+0.1")

## Plot for paper
current_gain = 0
Ioffset_fs = Iapp_fs
I_fs = [current_gain*valdata.I[i] .+ ones(1,length(valdata.I[i]))*Ioffset_fs for i=1:length(valdata.I)]
valdata_fs = IOData(valdata.V,Tuple(I_fs),valdata.T,valdata.t,Float32(dt_data))

# Plot
plt=getVoltagePlots(net_fs,valdata_fs;axisOpts...)
plt_fs_paper = plot(plt[1][1][:pltV̂];ylabel="",labelOpts...)
savefig(plt_fs_paper,string(figurepath,"PD-fast_removal.svg"))
plt_fs_paper