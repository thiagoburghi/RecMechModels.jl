## Load packages
include("./../../RecMechModels.jl")
datapath = string("examples/LP-PD/data/PD/")
modelpath = string("examples/LP-PD/results/models/PD/mechanistic/")
figurepath = string("examples/LP-PD/results/figures/")

## Load validation results
@load string(modelpath,"valTF.bson") valDict
@load string(modelpath,"valData.bson") valDataDict

# Parameters
n=m=1
sd=500.0
metric = :ValLoss
dt_data=0.1
Δ  = 0.001
β₁ = 0.9
β₂ = 0.99
cost_type=string(l2norm)
lastLayerWeight=1e3
chunk=1
batchsize_tf=192*200
Nepochs=200
ρexp = Dict("Experiment 22/PD1"=>1e-5,
            "Experiment 23/PD1"=>7.5e-5,
            "Experiment 24/PD1"=>5e-5,
            "Experiment 24/PD2"=>5e-5,
            "Experiment 25/PD1"=>2e-5)

####################################
## Paper Figure 4 C+D + SI Appendix 
####################################
# Load models for each experiment
expDict = Dict() 
for exp in keys(ρexp)
    constkeys = (exp,cost=string(cost_type),lastLayerWeight=lastLayerWeight,chunk=chunk,std=sd,Δ=Δ,β₁=β₁,β₂=β₂,ρ₀=ρexp[exp],batchsize=batchsize_tf,epochs=Nepochs)
    expDict[exp] = valDict[constkeys]
end

## Plot all losses
prepDict = Dict("Prep 1"=>expDict["Experiment 22/PD1"],"Prep 2"=>expDict["Experiment 23/PD1"],"Prep 3"=>expDict["Experiment 24/PD1"],"Prep 4"=>expDict["Experiment 24/PD2"],"Prep 5"=>expDict["Experiment 25/PD1"],)
bestHP,lossPlots = plotValLosses(prepDict,plotSize=(600,600),
                                    title="PD neuron training losses",
                                    hull=false,
                                    metric=:ValLoss,
                                    deltaEpochIndex=25,
                                    legend=:topright,
                                    plotBest=false,
                                    legendfontsize=10);
savefig(string(figurepath,"PD-losses.png"))

## Check IV curves
plotIV(expDict,plotSize=(800,600),
                    vrange=(-65,-10),
                    hyParsName="Prep")

## Simulate currents (single experiment)
experiment = "Experiment 22/PD1"
colorDict = Dict("Experiment 22/PD1"=>palette(:default)[1],
                 "Experiment 23/PD1"=>palette(:default)[2],
                 "Experiment 24/PD1"=>palette(:default)[3],
                 "Experiment 24/PD2"=>palette(:default)[4],
                 "Experiment 25/PD1"=>palette(:default)[5])
currentTicks = Dict("Experiment 22/PD1"=>(([5.0,0.0],["5.0nA","0.0nA"]),
                        ([0.2,0.1],["0.2mS","0.1mS"]),
                        ([0.09,0.05],["0.09mS","0.05mS"]),
                        ([0.11,0.09],["0.11mS","0.09mS"])),
                    "Experiment 23/PD1"=>(([4.0,-2.0],["4.0nA","-2.0nA"]),
                                    ([0.1,0.04],["0.1mS","0.04mS"]),
                                    ([0.09,0.05],["0.09mS","0.05mS"]),
                                    ([0.11,0.09],["0.11mS","0.09mS"])),
                    "Experiment 24/PD1"=>(([5.0,-2.5],["5.0nA","-2.5nA"]),
                                    ([0.15,0.1],["0.15mS","0.1mS"]),
                                    ([0.07,0.03],["0.07mS","0.03mS"]),
                                    ([0.02,0.01],["0.02mS","0.01mS"])),
                    "Experiment 24/PD2"=>(([3.0,-2.0],["3.0nA","-2.0nA"]),
                                    ([0.2,0.1],["0.2mS","0.1mS"]),
                                    ([0.06,0.02],["0.06mS","0.02mS"]),
                                    ([0.009,0.005],["0.009mS","0.005mS"])),
                    "Experiment 25/PD1"=>(([4.0,-2.0],["4.0nA","-2.0nA"]),
                                    ([0.08,0.04],["0.08mS","0.04mS"]),
                                    ([0.07,0.04],["0.07mS","0.04mS"]),
                                    ([0.045,0.02],["0.045mS","0.02mS"])))
condPlot=plotConductances(expDict[experiment][:netValLoss],valDataDict[experiment],
                    plotPercent=(0.225,0.6),  
                    # plotPercent=(0.225,0.35), #zoomed in
                    plotSize=(600,800),
                    tickfontsize=10,
                    xlabelfontsize=10,
                    ylabelfontsize=14,
                    predictionColor=colorDict[experiment],
                    vticks=([-20,-40],["-20mV","-40mV"]),
                    currentTicks=currentTicks[experiment],
                    overlay=false)
savefig(condPlot,string(figurepath,"conductances"*replace(experiment, "/" => "-")*".png"))

####################################
## Paper Figure 4 E (current removal)
####################################
# Choose experiment to plot 
experiment = "Experiment 22/PD1"
net_tf = expDict[experiment][:netValLoss]
valdata = valDataDict[experiment]

# A current ramp to catch oscillations
actvaldata=deepcopy(valdata)
actvaldata.V[1][1,:] = range(-80.0, -20.0; length=length(valdata.V[1]))
actvaldata.I[1][1,:] = 5*actvaldata.I[1][1,:]
actvaldata.I[1][1,:] += range(-30.0, 30.0; length=length(valdata.I[1]))

# Full model
Ioffset_full = 0.0
current_gain = 0.0
net_full = deepcopy(net_tf)
I_full = [current_gain*valdata.I[i] .+ ones(1,length(valdata.I[i]))*Ioffset_full for i=1:length(valdata.I)]
valdata_full = IOData(valdata.V,Tuple(I_full),valdata.T,valdata.t,Float32(dt_data))

plt=getVoltagePlots(net_full,valdata_full,
                    vlims=(-65,-15),
                    plotPercent=(0.8,1.0),
                    vticks=([-60,-40,-20],[L"-60mV",L"-40mV",L"-20mV"]),
                    tticks=([16,17],[L"1s",L"2s"]))
plt_full = plot(plt[1][1][:pltV̂],
                    legend=false,
                    framestyle=:grid,
                    ylabel="",
                    ytickfontsize=14,
                    xtickfontsize=14,
                    )
savefig(plt_full,string(figurepath,"PD-full_removal.png"))
plt_full

#######################################################
## Remove fast ionicCurrents and find bifurcations
#######################################################
net_su = deepcopy(net_tf)
net_su[1,1].ionicReadout[:] = [0.0,1.0,1.0,1.0]
bifs_su = findBifurcation(net_su,(1,1),Ω₀=[0.0,0.1,0.01,0.001],V₀=[-70,-65.0,-60.0,-55.0,-50,-45.])

##
plotVoltages(net_su,actvaldata,
                legend=false,
                tticks=false,
                plotPercent=(0.0,1.0))

## Simulate slow wave
# From bifurcation
bif_number = 3
v₀ = bifs_su[bif_number][:v]
Iapp₀ = bifs_su[bif_number][:Iapp]
# From eyeballed values
# v₀ = -60.0
# Iapp₀ = -14.0

# t_su,V_su₋ = net_su(v₀,Iapp₀,valdata.I[1],valdata.t)
# Probe around bifurcations
t_su,V_su₋ = net_su(v₀,Iapp₀-0.1,valdata.t)
t_su,V_su₊ = net_su(v₀,Iapp₀+0.1,valdata.t)
plt_su=plot(t_su/1000,V_su₋[1][:],label="Iapp-0.1")
plt_su=plot!(t_su/1000,V_su₊[1][:],label="Iapp+0.1")

## Plot for paper
current_gain = 0.0
Ioffset_su = Iapp₀
I_su = [current_gain*valdata.I[i] .+ ones(1,length(valdata.I[i]))*Ioffset_su for i=1:length(valdata.I)]
valdata_su = IOData(valdata.V,Tuple(I_su),valdata.T,valdata.t,Float32(dt_data))
plt=getVoltagePlots(net_su,valdata_su,
                    vlims=(-65,-15),
                    plotPercent=(0.8,1.0),
                    vticks=([-60,-40,-20],[L"-60mV",L"-40mV",L"-20mV"]),
                    tticks=([16,17],[L"1s",L"2s"]))
plt_su_paper = plot(plt[1][1][:pltV̂],
                    legend=false,
                    framestyle=:grid,
                    ylabel="",
                    ytickfontsize=14,
                    xtickfontsize=14,
                    )
savefig(plt_su_paper,string(figurepath,"PD-slow_removal.png"))
plt_su_paper

######################################################
## PD: Data with information about different α and σ
######################################################
data = Dict("Experiment 22/PD1"=>(id="Experiment 22/973_130_00",numbers=9:12,marker=:star,opacity=1.0,color=palette(:default)[1],
                number=Dict((α=0.015,σ=0.015) => 9, (α=0.01,σ=0.015) => 10, (α=0.005,σ=0.015) => 11, (α=0.001,σ=0.015) => 12,
                            (α=0.001,σ=0.03) => 13, (α=0.005,σ=0.03) => 14, (α=0.01,σ=0.03) => 15, (α=0.015,σ=0.03) => 16)),
            "Experiment 23/PD1"=>(id="Experiment 23/973_143_00",numbers=1:4,marker=:circle,opacity=1.0,color=palette(:default)[2],
                number=Dict((α=0.015,σ=0.015) => 1, (α=0.01,σ=0.015) => 2, (α=0.005,σ=0.015) => 3, (α=0.001,σ=0.015) => 4,
                            (α=0.001,σ=0.03) => 12, (α=0.005,σ=0.03) => 11, (α=0.01,σ=0.03) => 10, (α=0.015,σ=0.03) => 9)),
            "Experiment 23/PD2"=>(id="Experiment 23/973_143_00",numbers=17:20,marker=:square,opacity=1.0,color=palette(:default)[3],
                number=Dict((α=0.015,σ=0.015) => 17, (α=0.01,σ=0.015) => 18, (α=0.005,σ=0.015) => 19, (α=0.001,σ=0.015) => 20,
                            (α=0.001,σ=0.03) => 28, (α=0.005,σ=0.03) => 27, (α=0.01,σ=0.03) => 26, (α=0.015,σ=0.03) => 25)),
            "Experiment 24/PD1"=>(id="Experiment 24/973_143_1_00",numbers=1:4,marker=:diamond,opacity=1.0,color=palette(:default)[4],
                number=Dict((α=0.015,σ=0.015) => 1, (α=0.01,σ=0.015) => 2, (α=0.005,σ=0.015) => 3, (α=0.001,σ=0.015) => 4,
                            (α=0.001,σ=0.03) => 12, (α=0.005,σ=0.03) => 11, (α=0.01,σ=0.03) => 10, (α=0.015,σ=0.03) => 9)),
            "Experiment 24/PD2"=>(id="Experiment 24/973_143_1_00",numbers=21:24,marker=:triangle,opacity=1.0,color=palette(:default)[5],
                number=Dict((α=0.015,σ=0.015) => 21, (α=0.01,σ=0.015) => 22, (α=0.005,σ=0.015) => 23, (α=0.001,σ=0.015) => 24,
                            (α=0.001,σ=0.03) => 32, (α=0.005,σ=0.03) => 31, (α=0.01,σ=0.03) => 30, (α=0.015,σ=0.03) => 29)),
            "Experiment 25/PD1"=>(id="Experiment 25/973_146_00",numbers=21:24,marker=:utriangle,opacity=1.0,color=palette(:default)[6],
                number=Dict((α=0.015,σ=0.015) => 1, (α=0.01,σ=0.015) => 2, (α=0.005,σ=0.015) => 3, (α=0.001,σ=0.015) => 4,
                            (α=0.001,σ=0.03) => 12, (α=0.005,σ=0.03) => 11, (α=0.01,σ=0.03) => 10, (α=0.015,σ=0.03) => 9)))
