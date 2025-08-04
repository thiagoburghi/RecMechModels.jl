include("./../../RecMechModels.jl")
datapath = string("examples/LP-PD/data/LP-PD/")
modelpath = string("examples/LP-PD/results/models/LP-PD/")
figurepath = string("examples/LP-PD/results/figures/")

# Load all trained models
@load string(modelpath,"valTF.bson") dict_tf
@load string(modelpath,"valMS.bson") dict_ms

colorDict = Dict("file1" => palette(:default)[1],
                 "file2" => palette(:default)[2],
                 "file3" => palette(:default)[3])

# Hyperparameters of the model
dt=0.1
epochs_tf = 500
batchsize_tf = 192*200
ρ₀_tf = 0.0015
β₁_tf = 0.9
β₂_tf = 0.9

###############################################
## Figure 3 B
###############################################
# Choose model trained on specific dataset
filename = "file1"
net_tf = dict_tf[(filename=filename,ρ₀=ρ₀_tf,β₁=β₁_tf,β₂=β₂_tf,epochs=epochs_tf,batchsize=batchsize_tf)][:netValLoss]

# Load target data
_,valdata_all = loaddata(1,2,string(datapath,filename,".mat"), propTrainEnd=0.9, propVal=0.1,dt)
data=valdata_all[1]

# Simulate the model and recover synaptic current
t,V̂,X̂,V,Iapp = net_tf(data)
Isyn = net_tf[1,2](V̂[1],X̂[1,2])

inds = round(Int,length(t))-40000:round(Int,length(t))-20000
V̂_LP = V̂[1][inds]
V_LP = V[1][inds]
V_PD = V[2][inds]
Iapp = Iapp[1][inds]
t = t[inds]/1000
Isyn = Isyn[inds] .- minimum(Isyn[inds])

# Plot the results
vmax,vmin = maximum(V_LP)+1,minimum(V_LP)-1
v̂max,v̂min = maximum(V̂_LP)+1,minimum(V̂_LP)-1
plt1 = plot(t,V_PD, 
    xticks=(7:11, []),
    yticks=([-40,-20],["-40mv","-20mv"]),
    ylabel=string(L"v_{\mathrm{PD}}"," (input)"),
    color=:black)
plt2 = plot(t,Iapp, 
    xticks=(7:11, []),
    yticks=([-1.0,0.0,1.0],["-1.0nA","0.0nA","1.0nA"]),
    ylabel=string(L"I_{\mathrm{LP}}", " (input)"),
    color=:black)
plt3 = plot(t,V_LP, 
    xticks=(7:11, []),
    yticks=([-40,-30],["-40mv","-30mv"]),
    ylims=(vmin,vmax),
    ylabel=string(L"v_{\mathrm{LP}}", " (target)"),
    color=:black)
plt4 = plot(t,V̂_LP, 
    color=colorDict[filename],
    xticks=(7:11, []) , 
    ylims=(v̂min,v̂max),
    yticks=([-40,-30],["-40mv","-30mv"]),
    ylabel=string(L"v_{\mathrm{LP}}", " (pred.)"))
plt5 = plot(t,Isyn,
    color=colorDict[filename],
    xticks=(7:11, ["0s","1s","2s","3s"]),
    yticks=([0.0,1.0],["0.0nA","1.0nA"]),
    # ylims=(-0.1,1.2) ,
    ylabel=string(L"I_{\mathrm{syn}}", " (pred.)"))

l = @layout [a{0.25h}; b{0.125h}; c{0.25h}; d{0.25h}; e{0.125h}]
plt_val = plot(plt1,plt2,plt3,plt4,plt5,
    layout=l,
    framestyle=:grid,
    legend=false,
    bottom_margin=[-5Plots.mm -5Plots.mm -5Plots.mm -5Plots.mm 0Plots.mm],
    linewidth=2.0,
    size=(1000,1200))
plot!(plt_val, 
    ytickfontsize=16,
    xtickfontsize=16, 
    ylabelfontsize=20,
    left_margin=10Plots.mm,)

#
savefig(plt_val,string(figurepath,string("traces_LP-PD_",filename,".svg")))
plt_val

####################################################
## Figure 3 C
####################################################
# Plot TF validation curves
dict_tf_filenames = filterDict(dict_tf,(ρ₀=ρ₀_tf,β₁=β₁_tf,β₂=β₂_tf,epochs=epochs_tf,batchsize=batchsize_tf));
dict_tf_filenumber = Dict()
for filename in keys(dict_tf_filenames)
    number = parse(Int,split(filename,"file")[2])
    dict_tf_filenumber[number] = dict_tf_filenames[filename]
end
bestHP,plts_tf = plotValLosses(dict_tf_filenumber,lossType="Teacher forcing ",
                            hpName="Trial: ",
                            metric=:ValLoss,                # alternatively, :AngSep
                            plotBest=false,
                            deltaEpochIndex=25,
                            initEpochIndex=10,
                            opacityAfterFirst=0.6,
                            linewidth=3.0,
                            timeMode=:s)

## Plot TF vs MS validation curves
filename = "file1"

# Load TF and MS results
shotsize = 30
batchsize_ms = 192*200
epochs_ms = 500
snap_interval_ms = 5
ρ₀_ms = 0.0015
ρᵥ = 1.0
ρₓ = 0.1
Δ_ms = 0.003
β₁_ms = 0.9
β₂_ms = 0.9
std=200
val_tf = dict_tf[(filename=filename,ρ₀=ρ₀_tf,β₁=β₁_tf,β₂=β₂_tf,epochs=epochs_tf,batchsize=batchsize_tf)]
val_ms = dict_ms[(filename=filename,ρ₀=ρ₀_ms,ρₓ=ρₓ,Δ=Δ_ms,β₁=β₁_ms,β₂=β₂_ms,epochs=epochs_ms,batchsize=batchsize_ms,std=std,shotsize=shotsize)]

## Plot the validation losses
initEpochIndex = 1
deltaEpochIndex = 20
tTicks_ms = (val_ms[:epochs][initEpochIndex:deltaEpochIndex:end],[string("Epoch ",val_ms[:epochs][i]," \n", getTime(val_ms[:times][i])) for i=initEpochIndex:deltaEpochIndex:length(val_ms[:epochs])])

plt_ms = plot(val_tf[:epochs],log10.(val_tf[:valLoss]),
    opacity=0.6,
    color=palette(:default)[1],
    label="Teacher forcing",
    linewidth=3)
plt_ms = plot!(val_ms[:epochs] .+ val_tf[:epochs][val_tf[:minValLossInd]],log10.(val_ms[:valLoss]),
    xticks=(tTicks_ms[1].+val_tf[:epochs][val_tf[:minValLossInd]],tTicks_ms[2]),
    linewidth=3,
    color=palette(:default)[4],
    xtickfontcolor = palette(:default)[4],
    label="Multiple shooting",
    ylabel="log validation loss \n (trial 1)")

plt_loss_LP = plot(plts_tf...,plt_ms,layout=(length(plts_tf)+1,1),
                    size=(1000,900),
                    rightmargin=10Plots.mm,
                    ylabelfontsize=16,
                    xtickfontsize=14,
                    legendfontsize=14,
                    ytickfontsize=14)
display(plt_loss_LP)

## Save the plots
savefig(plt_loss_LP,string(figurepath,"losses_LP-PD.svg"))

###############################################
## FIGURE 3 D
## USE SAME FILE AND DATA AS IN FIGURE 3B
###############################################
# Choose best multiple shooting model and simulate
net_ms = val_ms[:netValLoss]
_,V̂ms,X̂ms = net_ms(data)

## Comparison figures
plt = plot(t,V_LP, 
    xticks=(7:11, []),
    yticks=([-40,-30,-20],[]),
    ylims=(vmin,vmax),
    xlims=(7.6,8.1),
    color=:black,
    linewidth=4.0,
    framestyle=:grid,
    opacity=0.4
    )
plt_tf = plot(plt,t,V̂_LP, 
    color=palette(:default)[1],
    linewidth=5.0)
plt_ms = plot(plt,t,V̂ms[1][inds],
    color=palette(:default)[4],
    linewidth=5.0)
plt_comparison = plot(plt_tf,plt_ms,
    layout=(2,1),
    legend=false,
    size=(1200,750),
    bottom_margin=[-15Plots.mm 0Plots.mm])

## Save the plots
savefig(plt_comparison,string(figurepath,"traces_LP-PD_TFvsMS.svg"))