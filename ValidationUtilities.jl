# 
function processVoltage(v,prefilter,threshold,kernel,window)
    # Bandpass filtering
    if !isnothing(prefilter)
        # Forward and backward application of the filter (non-causal, doubles order)
        v̂_f = filtfilt(prefilter,v)   
        # Apply symmetrized filter(non-causal, preserves order, assumes DC=0)
        # v̂_f = filt(prefilter,v)
        # v̂_f += reverse(filt(prefilter,reverse(v)))
    else
        v̂_f = v
    end
    # Threshold to isolate super-threshold part
    if threshold > -Inf
        v̂_f = relu.(v̂_f .- threshold)
    end
    if window > 0
        spk_inds=findmaxima(v̂_f,window)[:indices]
        v̂_f = zeros(length(v̂_f))
        v̂_f[spk_inds] .= 1.0
    end
    # Smooth it out
    if !isnothing(kernel)
        # for non-causal kernels we should technically offset by length(kernel)
        # correct for that when computing the cost
        offset = fld(length(kernel),2)
        v̂_f = DSP.conv(v̂_f,kernel)
        v̂_f = v̂_f[offset+1:end-offset]
    end
    return v̂_f
end

function smoothingKernel(kernelType, std)
    cutoff = round(Int, 6.0 * std)
    k = cutoff:-1:-cutoff
    if kernelType==:gaussian
        kernel = exp.(-k.^2 / (2 * std^2))
        kernel /= sum(kernel)
    elseif kernelType==:laplace
        kernel = exp.(-abs.(k) / std)
        kernel /= sum(kernel)
    else
        error("Kernel type not recognized. Possible values are :gaussian and :laplace")
    end
end

##
# experiment="Experiment 22/PD1"
# v = expDict[experiment][:valDict][expDict[experiment][:bestHP][1]][:data][1][1][:]
# v_ang_sep = expDict[experiment][:valDict][expDict[experiment][:bestHP][1]][:predictionAngSep][1][1][:]
# prefilter=(fc_low=1/50,fc_high=1/2,order=3)
# prefilter = digitalfilter(Bandpass(prefilter[:fc_low], prefilter[:fc_high], fs=1/0.1), Butterworth(prefilter[:order]))
# kernel = smoothingKernel(:gaussian, 1000.0)
# thr= 5.
# window = 10
# vf = processVoltage(v,prefilter,thr,kernel,window)
# vf_ang_sep = processVoltage(v_ang_sep,prefilter,thr,kernel,window)
# # plot(v_ang_sep)
# plt1=plot(vf)
# plot!(vf_ang_sep)
# plt2=plot(v)
# plt3=plot(v_ang_sep)
# plot(plt1,plt2,plt3,layout=(3,1))
# xlims!(50000,90000)
##

# Must change to allow validation with multiple datasets
function validate(snapshots,valdata::IOData; prefilter::Union{NamedTuple,Nothing}=(fc_low=1/10,fc_high=1/2,order=6),
                                             threshold::Real=-Inf,
                                             smooth::Union{NamedTuple}=(kernelType=:gaussian,std=50.0),
                                             window=0)    
    # Some admin
    if !isnothing(prefilter)
        prefilter = digitalfilter(Bandpass(prefilter[:fc_low], prefilter[:fc_high], fs=1/valdata.dt), Butterworth(prefilter[:order]))
        offset = 5000
    end
    if !isnothing(smooth)
        kernel = smoothingKernel(smooth[:kernelType], smooth[:std])
        offset = fld(length(kernel),2)
    else
        kernel = nothing
        offset = 0
    end

    # Validation data
    # COMPUTE THIS IN THE DISTRIBUTED LOOP IN CASE WE LEARN TIMESCALES
    RNNvaldata = MSData(snapshots[1][:model],valdata)
    n_neurons = snapshots[1][:model].size[1]

    V = [RNNvaldata.rawdata[1].V[n][:] for n=1:n_neurons]
    V_f = [processVoltage(V[n],prefilter,threshold,kernel,window) for n=1:n_neurons]
    data_traces = (unfiltered=V,filtered=V_f)

    epochs = SharedArray{Int}(length(snapshots))
    times = SharedArray{Float64}(length(snapshots))
    ol_loss = SharedArray{Float64}(length(snapshots))
    cl_loss = SharedArray{Float64}(length(snapshots))
    ang_sep = SharedArray{Float64}(length(snapshots))
    predicted_traces = @DArray [deepcopy(data_traces) for _=1:length(snapshots)]

    @sync @distributed for i=1:length(snapshots)
        # Load model
        net = snapshots[i][:model]

        # Validation data
        # UNCOMMENT THIS IF WE LEARN TIMESCALES
        # RNNvaldata = MSData(net,valdata)

        # Simulate model
        V̂seq,X̂seq = net(RNNvaldata)
        V̂,X̂ = shotTrajectories(V̂seq,X̂seq)
        V̂ = [V̂[1][n][:] for n=1:n_neurons]  # first index is batch number
        V̂_f = [processVoltage(V̂[n],prefilter,threshold,kernel,window) for n=1:n_neurons]
        for n=1:n_neurons
            predicted_traces[i][:unfiltered][n][:] = V̂[n]
            predicted_traces[i][:filtered][n][:] = V̂_f[n]
        end

        # Compute closed-loop MSE
        cl_loss[i] = sum([Flux.mse(V̂_f[n],V_f[n]) for n=1:n_neurons])/n_neurons
        ang_sep[i] = sum([angular_separation(V̂_f[n][1+offset:end-offset],V_f[n][1+offset:end-offset]) for n=1:n_neurons])/n_neurons 
        ol_loss[i] = snapshots[i][:loss]
        epochs[i] = snapshots[i][:epoch]
        times[i] = snapshots[i][:time]
        println(string("Simulated model number ",i," of ",length(snapshots)))
    end
    predicted_traces = collect(predicted_traces)
    data_traces = collect(data_traces)

    cl_loss[isnan.(cl_loss)] .= Inf
    minValLoss, min_valLoss_ind = findmin(cl_loss)
    netValLoss = snapshots[min_valLoss_ind][:model]
    predictionValLoss = predicted_traces[min_valLoss_ind]

    ang_sep[isnan.(ang_sep)] .= 0.0
    maxAngSep, max_angSep_ind = findmax(ang_sep)
    netAngSep = snapshots[max_angSep_ind][:model]
    predictionAngSep = predicted_traces[max_angSep_ind]
    
    println(string("Minimum validation loss: epoch ",epochs[min_valLoss_ind],"; mse: ", minValLoss))
    println(string("Maximum angular separation: epoch ",epochs[max_angSep_ind],"; value: ", maxAngSep))

    return (netValLoss=netValLoss,predictionValLoss=predictionValLoss,minValLossInd=min_valLoss_ind,
            netAngSep=netAngSep,predictionAngSep=predictionAngSep,maxAngSepInd=max_angSep_ind,
            netLastEpoch=snapshots[end][:model],predictionLastEpoch=predicted_traces[end],lastEpochInd=length(snapshots),
            data=data_traces,valLoss=cl_loss,angSep=ang_sep,trainLoss=ol_loss,epochs=epochs,times=times)
end

"""
    Reducing size of dicts for analysis
"""
# function buildDict!(valDict,hpDict,hps; metric=:ValLoss, optMetric=Inf, hpsVals)
#     if length(hps) > 0
#         for hpval = (isnothing(hpvals[1]) ? unique(key[hps[1]] for key in keys(valDict)) : hpvals[1])
#             hpDict[hps[1]] = hpval
#             buildDict!(valDict,hpDict,hps[2:end]; metric=metric, optMetric=optMetric, hpsVals=hpsVals[2:end])
#         end
#     else
#         key = (;(key => hpDict[key] for key in keys(first(keys(valDict))))...)
#         # Filter bad models out!
#         if haskey(valDict,key)
#             if metric == :angSep
#                 angSep = valDict[key][:angSep]
#                 maxAngSepInd = valDict[key][:maxAngSepInd]
#                 if angSep[maxAngSepInd] > optMetric
#                     dict[NamedTuple{Tuple([hp1,hp2])}((hp1val,hp2val))] = valDict[key]
#                 end
#             elseif metric == :valLoss
#                 valLoss = valDict[key][:valLoss]
#                 minValLossInd = valDict[key][:minValLossInd]
#                 if log10(valLoss[minValLossInd]) < optMetric
#                     dict[NamedTuple{Tuple([hp1,hp2])}((hp1val,hp2val))] = valDict[key]
#                 end
#             else
#                 error("Metric not recognized. Possible values are :AngSep and :ValLoss")
#             end
#             modelcount += 1
#     end
# end

function reduceDict(valDict::Dict, hpDict::Dict, hp1::Symbol, hp2::Symbol; metric=:ValLoss, optMetric=Inf, reduce=0, hp1vals=nothing, hp2vals=nothing)
    dict=Dict()
    modelcount = 0
    metric = Symbol(lowercasefirst(string(metric)))
    # length(hpsVals) == 0 ? hpsVals = [nothing for _ in 1:length(hps)] : (length(hpsVals) == length(hps) ? nothing : error("Number of hyperparameter values must match number of hyperparameters"))

    for hp1val = (isnothing(hp1vals) ? unique(key[hp1] for key in keys(valDict)) : hp1vals)
        hpDict[hp1] = hp1val
        for hp2val = (isnothing(hp2vals) ? unique(key[hp2] for key in keys(valDict)) : hp2vals)
            hpDict[hp2] = hp2val
            key = (;(key => hpDict[key] for key in keys(first(keys(valDict))))...)
            # Filter bad models out!
            if haskey(valDict,key)
                if metric == :angSep
                    angSep = valDict[key][:angSep]
                    maxAngSepInd = valDict[key][:maxAngSepInd]
                    if angSep[maxAngSepInd] > optMetric
                        dict[NamedTuple{Tuple([hp1,hp2])}((hp1val,hp2val))] = valDict[key]
                    end
                elseif metric == :valLoss
                    valLoss = valDict[key][:valLoss]
                    minValLossInd = valDict[key][:minValLossInd]
                    if log10(valLoss[minValLossInd]) < optMetric
                        dict[NamedTuple{Tuple([hp1,hp2])}((hp1val,hp2val))] = valDict[key]
                    end
                else
                    error("Metric not recognized. Possible values are :AngSep and :ValLoss")
                end
                modelcount += 1
            end
        end  
    end
    if reduce>0
        rev = (metric==:angSep ? true : false)
        optMetricInd = (metric==:angSep ? :maxAngSepInd : :minValLossInd)
        bestHP = sort(collect(keys(dict)), by = hp -> dict[hp][metric][dict[hp][optMetricInd]], rev=rev)
        println(bestHP)
        for hp in bestHP[end-(reduce-1):end]
            pop!(dict,hp)
        end
    end
    println("\nNumber of good models: ",length(dict)," out of ",modelcount,"\n")  
    return dict
end

"""
    Functions for plotting validation results
"""
function plotValLosses(val::NamedTuple;metric=:AngSep,kwargs...)
    dict=Dict()
    dict["Validation"] = val
    bestHP,_ = plotValLosses(dict;plotBest=false,metric=metric,kwargs...)
    netType = Symbol("net" * string(metric))
    return dict[bestHP[1]][netType]
end

function plotValLosses(valDict::Dict;   title::String="",
                                        hpName="",
                                        lossType="",
                                        plotSize=(600,800),
                                        metric=:AngSep,
                                        plotBest=true,
                                        deltaEpochIndex=10,
                                        initEpochIndex=1,
                                        opacityAfterFirst=1.0,
                                        legendfontsize=12,
                                        ylabelfontsize=12,
                                        linewidth=2.0,
                                        legend=true,
                                        epochTimes=true,
                                        timeMode=:ms)
    plts = plotBest ? [plot() for _ in 1:4] : [plot() for _ in 1:2]
    local bestHP
    opct = 1.0
    for hp in sort(collect(keys(valDict)))
        valLoss = valDict[hp][:valLoss]
        angSep = valDict[hp][:angSep]
        epochs = valDict[hp][:epochs]

        # Get times to print in x tick labels
        trainingTicks = (epochs[initEpochIndex:deltaEpochIndex:end],["" for i=initEpochIndex:deltaEpochIndex:length(epochs)])
        if epochTimes
            tTicks = (epochs[initEpochIndex:deltaEpochIndex:end],[string("Epoch ",epochs[i]," \n", getTime(valDict[hp][:times][i];mode=timeMode)) for i=initEpochIndex:deltaEpochIndex:length(epochs)])
        else
            tTicks = (epochs[initEpochIndex:deltaEpochIndex:end],[string("Epoch ",epochs[i]) for i=initEpochIndex:deltaEpochIndex:length(epochs)])
        end

        # Training loss plot
        plot!(plts[1],epochs,log10.(valDict[hp][:trainLoss]);
                                    title=title,
                                    ylabel=string(lossType,"\n log training loss"),
                                    label=string(hpName,hp),
                                    legend=legend,
                                    xticks=trainingTicks,
                                    legendfontsize=legendfontsize,
                                    linewidth=linewidth,
                                    opacity=opct)
        
        commonOpts = (xticks=tTicks,legend=false,linewidth=linewidth,opacity=opct)
        
        # Validation plots
        if metric == :AngSep
            plot!(plts[2],epochs,angSep;
                            ylabel=string(hull ? "max " : "","validation \n angular separation"),
                            ylims=(0.0,1.01),
                            commonOpts...)
            bestHP = sort(collect(keys(valDict)), by = hp -> valDict[hp][:angSep][valDict[hp][:maxAngSepInd]], rev=true)
        elseif metric == :ValLoss
            plot!(plts[2],epochs,log10.(valLoss);
                            ylabel=string(lossType,"\n log validation loss"),
                            commonOpts...)
            bestHP = sort(collect(keys(valDict)), by = hp -> valDict[hp][:valLoss][valDict[hp][:minValLossInd]], rev=false)
        else
            error("Metric not recognized. Possible values are :AngSep and :ValLoss")        
        end
        opct = opacityAfterFirst
    end

    commonOpts = (xlabel=hpName,marker=:circle,legend=false)
    if plotBest
        if metric == :AngSep
            best_angsep = [valDict[hp][:angSep][valDict[hp][:maxAngSepInd]] for hp in sort(collect(keys(valDict)))]  
            best_times = [valDict[hp][:times][valDict[hp][:maxAngSepInd]] for hp in sort(collect(keys(valDict)))]
            # Best results
            plot!(plts[3],sort(collect(keys(valDict))),best_angsep;
                                            title="Best model angular separation",
                                            commonOpts...)
            # Corresponding times
            plot!(plts[4],sort(collect(keys(valDict))),best_times;
                                            title="Best model training time",
                                            yticks=(best_times,[getTime(t;mode=timeMode) for t in best_times]),
                                            commonOpts...)
        else
            best_losses = [valDict[hp][:valLoss][valDict[hp][:minValLossInd]] for hp in sort(collect(keys(valDict)))]
            best_times = [valDict[hp][:times][valDict[hp][:minValLossInd]] for hp in sort(collect(keys(valDict)))]
            # Best results
            plot!(plts[3],sort(collect(keys(valDict))),best_losses;
                                            title="Best model validation loss",
                                            commonOpts...)
            # Corresponding times
            plot!(plts[4],sort(collect(keys(valDict))),best_times;
                                            title="Best model training time",
                                            yticks=(best_times,[getTime(t;mode=timeMode) for t in best_times]),
                                            commonOpts...)
        end
    end

    # Some text output
    if metric == :AngSep
        println("Best hyperparameters in decreasing order: ")
        println(Vector{Any}(bestHP))
        println("Angular separation: ")
        println([valDict[hp][:angSep][valDict[hp][:maxAngSepInd]] for hp in bestHP])
        println("Corresponding log10 MSE loss: ")
        println([log10(valDict[hp][:valLoss][valDict[hp][:maxAngSepInd]]) for hp in bestHP])
        println("Corresponding epochs: ")
        println([valDict[hp][:epochs][valDict[hp][:maxAngSepInd]] for hp in bestHP])
        println("Corresponding times: ")
        println([getTime(valDict[hp][:times][valDict[hp][:maxAngSepInd]];mode=timeMode) for hp in bestHP])
    else
        println("Best hyperparameters in decreasing order: ")
        println(Vector{Any}(bestHP))
        println("log10 MSE loss: ")
        println([log10(valDict[hp][:valLoss][valDict[hp][:minValLossInd]]) for hp in bestHP])
        println("Corresponding angular separation: ")
        println([valDict[hp][:angSep][valDict[hp][:minValLossInd]] for hp in bestHP])
        println("Corresponding epochs: ")
        println([valDict[hp][:epochs][valDict[hp][:minValLossInd]] for hp in bestHP])
        println("Corresponding times: ")
        println([getTime(valDict[hp][:times][valDict[hp][:minValLossInd]];mode=timeMode) for hp in bestHP])
    end

    plt = plot(plts...,layout=(length(plts),1),size=plotSize,ylabelfontsize=ylabelfontsize,leftmargin=5Plots.mm,bottommargin=5Plots.mm)
    display(plt)
    return bestHP,plts
end

function getTime(t::Float64; mode=:ms)
    t = round(Int,t)
    if mode == :hms
        h = floor(Int,t/3600)
        m = floor(Int,(t-h*3600)/60)
        s = t-h*3600-m*60
        return string(h,"h ",m,"m ",s,"s")
    elseif mode ==:ms
        m = floor(Int,t/60)
        s = t-m*60
        return string(m,"m ",s,"s")
    elseif mode==:s
        return string(t," seconds")
    else
        error("Mode not recognized. Possible values are :hms, :ms, and :s")
    end
end

function plotValVoltages(val::NamedTuple; 
                    title::String="Title",
                    plotPercent::Tuple{AbstractFloat,AbstractFloat}=(0.0,1.0),
                    vlim=:auto,
                    plotSize=(600,400),
                    filtered=false,
                    tlabel="time [s]",
                    vLabel=true,
                    overlay=true,
                    vLegend=true,
                    vTicks=([-50,-30],["-50","-30"]),
                    legend=true,
                    metric=:AngSep,
                    linewidth=1.5,
                    downsample=1)

    n_neurons=length(val[:data][1])
    L = length(val[:data][1][1])
    inds = floor(Int,L*plotPercent[1])+1:downsample:floor(Int,L*plotPercent[2])
    L = length(inds)
    netType = Symbol("net" * string(metric))
    predictionMetric = Symbol("prediction" * string(metric))
    t = (1:L)*val[netType].cell.dt/1000

    commonOpts = (linewidth=linewidth,yticks=vTicks,xticks=false,ylim=vlim,xlabel="")

    plts=[]
    for n=1:n_neurons
        plt1=plot(t,val[:data][1][n][inds],
            label = vLegend ? string("target") : false,
            ylabel=(isa(vLabel,Bool) ? (vLabel ? L"$v_%$n$ [mV]" : "") : vLabel[n]),
            title=(n==1 ? title : ""),
            legend=(n==1 ? legend : false);
            commonOpts...
            )
        plt2 = overlay ? plt1 : plot()
        plt2 = plot!(plt2,t,val[predictionMetric][1][n][inds],
            label = vLegend ? string("prediction") : false,
            ylabel=(isa(vLabel,Bool) ? (vLabel ? L"$v_%$n$ [mV]" : "") : vLabel[n]),
            color=palette(:default)[2],
            legend=(n==1 ? legend : false);
            commonOpts...)
        overlay ? push!(plts,plt2) : append!(plts,[plt1,plt2])
        if filtered
            plt3=plot(t,val[:data][2][n][inds],label=string("Data ",n),ylabel="Filtered traces")
            push!(plts,plot(plt3,t,val[predictionMetric][2][n][inds],label=string("Filtered ",n)))
        end
        # Plot time info only on bottommost plot
        if n==n_neurons
            if isa(tlabel,String)
                plot!(plts[end],xlabel=tlabel,xticks=true)
            elseif isa(tlabel,Real)
                plot!(plts[end],[0.0,tlabel],ylims(plts[end])[1]*[1,1],color=:black,linewidth=3,label=false)
                annotate!(0.5, ylims(plts[end])[1],text("$tlabel s",:top,Plots.default(:fontfamily)))
            else
                error("tlabel must be a String or a Real number")
            end
        end
    end
    plt = plot(plts..., leftmargin=5Plots.mm,
                        bottommargin=hcat([-2.5Plots.mm for i=1:1, j_=1:length(plts)-1],[5Plots.mm;;]),
                        layout=(length(plts),1),
                        size=plotSize)
    display(plt)
    return plt
end

function plotValVoltages(valDict::Dict, hpList; plotSize=(800,600),kwargs...)
    pltList=[]
    eachPlotSize = plotSize./length(hpList)
    for hp in hpList
        plt=plotValVoltages(valDict[hp];title=string(hp),plotSize=eachPlotSize,kwargs...)
        push!(pltList,plt)
    end
    if length(pltList)==1
        plot(pltList[1],size=plotSize)    
    else
        iseven(length(pltList)) ? nothing : push!(pltList,plot())
        plot(pltList...,layout=(round(Int,length(pltList)/2),2),size=plotSize)
    end
end

"""
    Plots intrinsic current IV curves of a model, and optionally of a dataset.
    Keyword arguments:
    n: number of the neuron in the network
    data: NamedTuple or Dictionary containing IV curves of the leak and ionic currents in the data
"""
function plotIV(net::Network; n = 1,                # neuron number
                              m = 1,                # total current number
                              vrange=(-65,0.0),
                              plotSize=(800,600),
                              data=nothing)
    # Recover current names
    currentNames = keys(net[n,m].ionicCurrents)
    n_currents = length(currentNames)

    # Compute IV curves
    V̄ = vrange[1]:0.01:vrange[2]
    IVion,IVleak=IV(net,V̄,V̄)

    # Initiate plots
    plt_tot,plt_leak,plt_ion = plot(),plot(),[plot() for _ in 1:length(currentNames)]

    # Plot IV curves
    plotIV!(plt_tot,plt_leak,plt_ion,V̄,IVleak[n],IVion[n,m],currentNames;lb="Model")
    if !isnothing(data)
        plotIV!(plt_tot,plt_leak,plt_ion,V̄,data[:iLeak],data[:iIon],currentNames;lb="Data")
    end
    iseven(n_currents) ? nothing : push!(plt_ion,plot())
    plot(plt_tot,plt_leak,plt_ion...,
        layout=(2,ceil(Int,(n_currents+2)/2)), 
        size=plotSize,
        margin=5.0Plots.mm)
end

function plotSSActivations(net::Network; n = 1,                # neuron number
                                         m = 1,                # total current number
                                         k=1,
                                         vrange=(-80,0.0),
                                         plotSize=(800,600),
                                         actLims=(0,1))
    # Recover current names
    currentNames = keys(net[n,m].ionicCurrents)
    n_currents = length(currentNames)

    # Compute IV curves
    V̄ = vrange[1]:0.01:vrange[2]
    ssAct = ssActivations(net,V̄,V̄)

    plts=[]
    for j=1:length(ssAct[n,m])
        plt=plot()
        if !isnothing(ssAct[n,m][j][1]) 
            plt = plot!(plt,V̄,ssAct[n,m][j][1][k,:],title=currentNames[j],label="activation")
        end
        if !isnothing(ssAct[n,m][j][2])
            plt = plot!(plt,V̄,ssAct[n,m][j][2][k,:],title=currentNames[j],label="inactivation")
        end
        if !isnothing(ssAct[n,m][j][1]) || !isnothing(ssAct[n,m][j][2])
            push!(plts,plt)
        end
    end
    plot(plts...,layout=(length(plts),1),size=(800,800),ylims=actLims)
end

"""
    Plots IV curves for a dictionary of validation results.
    Useful for comparing hyperparameters.
    Keyword arguments:
    n: number of the neuron in the network
"""
function plotIV!(valDict::Dict, IVplts=nothing; n = 1,m = 1,
                                                envelope=false,
                                                hyParsName::Union{Nothing,String}=nothing,
                                                vrange=(-65,-20),
                                                plotSize=(800,600),
                                                vticks=:auto,
                                                metric=:AngSep,
                                                color=:blue)
    # Some admin
    local plotLeak
    V̄ = vrange[1]:0.01:vrange[2]
    netType = Symbol("net" * string(metric))
    
    # Recover current names
    currentNames = keys(valDict[first(valDict)[1]][netType][n,m].ionicCurrents)
    n_currents = length(currentNames)

    isnothing(hyParsName) ? hyParsName = "Hyperparameter" : nothing

    # Initialize plots
    if isnothing(IVplts)
        plt_tot,plt_leak,plt_ion = plot(),plot(),[plot() for _ in 1:n_currents]
    else
        plt_tot = IVplts[1]
        plt_leak = IVplts[2]
        plt_ion = IVplts[3]
    end

    # Plot IV curves for the network associated to each hyperparameter
    if !envelope
        for hp in sort(collect(keys(valDict)))
            # Recover model
            net = valDict[hp][netType]
            # Compute IV curves
            IVion,IVleak=IV(net,V̄,V̄)
            plotIV!(plt_tot,plt_leak,plt_ion,V̄,IVleak[n],IVion[n,m],vticks=vticks,currentNames,lb="$hyParsName = $hp")
            isnothing(IVleak[n]) ? plotLeak = false : plotLeak = true
        end
    else
        IVleak_all = zeros(length(valDict),length(V̄))
        IVion_all = zeros(n_currents,length(valDict),length(V̄))
        for (i,hp) in enumerate(sort(collect(keys(valDict))))
            # Recover model
            net = valDict[hp][netType]
            # Compute IV curves
            IVion,IVleak=IV(net,V̄,V̄)
            for j=1:n_currents
                IVion_all[j,i,:] = IVion[n,m][j,:]
            end
            if !isnothing(IVleak[n])
                IVleak_all[i,:] = IVleak[n][:]
                plotLeak = true
            else
                plotLeak = false
            end
        end

        # Plot mean and standard deviation of IV curves
        if plotLeak
            IVleak_mean = mean(IVleak_all,dims=1)[:]
            IVleak_std = std(IVleak_all,dims=1)[:]
        else
            IVleak_mean=IVleak_std=nothing
        end
        IVion_mean = dropdims(mean(IVion_all,dims=2),dims=2)
        IVion_std = dropdims(std(IVion_all,dims=2),dims=2)

        plotIV_envelope!(plt_tot,plt_leak,plt_ion,V̄,IVleak_mean,IVion_mean,IVleak_std,IVion_std,currentNames,vticks=vticks,color=color,lb=hyParsName)
    end
    if plotLeak
        plts = (iseven(n_currents) ? (plt_tot,plt_leak,plt_ion...) : (plt_tot,plt_leak,plt_ion...,plot()))
    else
        plts = (isodd(n_currents) ? (plt_tot,plt_ion...,plot()) : (plt_tot,plt_ion...))
    end
    plt=plot(plts...,
        layout=(2,ceil(Int,(n_currents+2)/2)), 
        size=plotSize,
        margin=5.0Plots.mm)
    return plt
end

function plotIV!(plt_tot,plt_leak,plt_ion,V̄,IVleak,IVion,currentNames;lb="label",vticks=:auto)
    IVtot = sum(IVion,dims=1)[:]

    opts = (lw=2,label=lb,xlabel=L"$v$ [mV]",xticks=vticks)
    # Plot Leak curves
    if !isnothing(IVleak)
        IVtot += IVleak[:]
        plt_leak=plot!(plt_leak,V̄[:],IVleak[:],
                                    legend=false,
                                    ylabel="[nA]",
                                    title="Leak current";
                                    opts...)
    end

    # Plot Ion ionicCurrents
    for i=1:length(currentNames)
        plt_ion[i]=plot!(plt_ion[i],V̄[:],IVion[i,:],
                                legend=false,
                                ylabel=(i==1 ? "[nA]" : ""),
                                title=string(currentNames[i]," current");
                                opts...)
    end
    # Plot Total current
    plt_tot=plot!(plt_tot,V̄[:],IVtot[:],
                                legend=:topright,
                                ylabel="[nA]",
                                title="Total current";
                                opts...)
end

function plotIV_envelope!(plt_tot,plt_leak,plt_ion,V̄,IVleak,IVion,IVleak_std,IVion_std,currentNames;vticks=:auto,color=:blue,lb="")
    # Make sure standard deviaton is computed properly
    α=0.3
    mean_opts = (lw=2,xticks=vticks,label=lb*" mean",c=color,xlabel=L"$v$ [mV]")
    std_opts = (fillalpha=α,lw=0,c=color,label="±1 SD")

    if !isnothing(IVleak)
        IVtot = sum(IVion,dims=1)[:] + IVleak[:]
        IVtot_std = sqrt.(sum(IVion_std.^2, dims=1)[:] + IVleak_std.^2)
        # Plot Leak curves
        plt_leak=plot!(plt_leak,V̄[:],IVleak[:],
                                    legend=false,
                                    title="Leak current";
                                    mean_opts...)

        plt_leak=plot!(plt_leak,V̄[:],IVleak[:].+IVleak_std[:],fillrange=(IVleak[:].-IVleak_std[:]);std_opts...)
    else
        IVtot = sum(IVion,dims=1)[:]
        IVtot_std = sqrt.(sum(IVion_std.^2, dims=1)[:])
    end
    
    # Plot Ion ionicCurrents
    for i=1:length(currentNames)
        plt_ion[i]=plot!(plt_ion[i],V̄[:],IVion[i,:];
                                legend=(i==1 ? :bottomleft : false),
                                title=string(currentNames[i]," current"),
                                mean_opts...)
        plt_ion[i]=plot!(plt_ion[i],V̄[:],IVion[i,:].+IVion_std[i,:],fillrange=(IVion[i,:].-IVion_std[i,:]);std_opts...)
    end
    # Plot Total current
    plt_tot=plot!(plt_tot,V̄[:],IVtot[:],
                                legend=:topright,
                                title="Total current";
                                mean_opts...)
    plt_tot=plot!(plt_tot,V̄[:],IVtot[:]+IVtot_std[:],fillrange=IVtot[:]-IVtot_std[:];std_opts...)
end

"""
    Function used to plot prediction vs data for a model
"""
function getVoltagePlots(net::Network,data::IOData; neurons::AbstractVector{Int}=[1,],
                                                    plotPercent::Tuple{AbstractFloat,AbstractFloat}=(0.0,1.0),
                                                    vlims=:auto,
                                                    vticks=:auto,
                                                    Iticks=:auto,
                                                    tbar=nothing,   # time bar in seconds
                                                    tticks=:auto,
                                                    tlabel="t [ms]",    
                                                    IUnit="[nA]",
                                                    overlay=false,
                                                    linewidth=1.5,
                                                    predictionColor=palette(:default)[1],
                                                    vtitle="",
                                                    gtf=false,
                                                    downsample=1,
                                                    tlabelfontsize=12)
    # Simulate the network
    if !gtf 
        t,V̂,X̂,V,Iapp,T = net(data)
    else
        t,V̂,X̂,V,Iapp,T = teacher(net,data)
    end

    # Get the indices to plot
    L = length(t)
    inds = floor(Int,L*plotPercent[1])+1:downsample:floor(Int,L*plotPercent[2])

    # Type of time axis
    if !isnothing(tbar)
        tticks = false
        tlabel = ""
    end

    # Plot the predictions and the voltage
    plt_neurons = []
    for n in neurons
        plt1 = plot(t[inds]/1000,V[n][inds],xlabel=tlabel,
                                            xticks=tticks,
                                            yticks=vticks,
                                            ylims=vlims,
                                            color=:black,
                                            linewidth=linewidth,
                                            ylabel=L"$v_{\mathrm{target}}$ [mV]",
                                            title=vtitle)        
        plt2 = plot(t[inds]/1000,V̂[n][inds],xlabel="",
                                            xticks=tticks,
                                            yticks=vticks,
                                            ylims=vlims,
                                            linewidth=linewidth,
                                            color=predictionColor,
                                            ylabel=L"$v$ [mV]")
        if overlay
            plt2 = plot!(t[inds]/1000,V[n][inds],xlabel=tlabel,color=:black,linewidth=linewidth,opacity=0.5,style=:dash,ylabel="")
        end
        plt3 = plot(t[inds]/1000,Iapp[n][inds],xticks=tticks,yticks=Iticks,color=:black,ylabel=L"$I_{\mathrm{app}}$ %$IUnit")
        if !isnothing(tbar)
            for p = [plt1,plt2,plt3]
                plot!(p,[t[inds[1]]/1000,t[inds[1]]/1000+tbar],ylims(p)[1]*[1,1],color=:black,linewidth=linewidth,label=false)
                annotate!(t[inds[1]]/1000+tbar/2, ylims(p)[1],text("$tbar s",:top,Plots.default(:fontfamily),tlabelfontsize))
            end
        end
        push!(plt_neurons,(pltV=plt1,pltV̂=plt2,pltIapp=plt3))
    end
    return plt_neurons,t,V̂,X̂,V,Iapp,T,inds
end

function plotVoltages(net::Network,data::IOData; neurons::AbstractVector{Int}=[1,], 
                                            legend=true,
                                            plotSize=(800,600),
                                            tickfontsize=12,
                                            labelfontsize=14,
                                            overlay=false, kwargs...)
    plt_neurons,_ = getVoltagePlots(net,data; neurons=neurons, overlay=overlay, kwargs...)
    plts=[]
    if overlay 
        for n=neurons
            l = @layout [a{0.8h}; b{0.2h}]
            push!(plts,plot(plt_neurons[n].pltV̂,plt_neurons[n].pltIapp,layout=l,size=plotSize))
        end
    else
        for n=neurons
            l = @layout [a{0.425h}; b{0.425h}; c{0.15h}]
            push!(plts,plot(plt_neurons[n].pltV,plt_neurons[n].pltV̂,plt_neurons[n].pltIapp,layout=l,size=plotSize))
        end
    end
    plot(plts...,layout=(1,length(plts)),legend=legend,labelfontsize=labelfontsize,tickfontsize=tickfontsize,leftmargin=5Plots.mm,bottommargin=5Plots.mm)
end

function plotConductances(net::Network,iodata::IOData; kwargs...)
    plotCurrents(net,iodata; plotConductances=true, kwargs...)
end

function plotCurrents(net::Network,iodata::IOData;  n=1,    #neuron number
                                                    m=1,    #total current number
                                                    legend=false,
                                                    tbar=nothing,   # time bar in seconds
                                                    tticks=:auto,
                                                    tlabel="t [s]",
                                                    gUnit="[mS]",
                                                    IUnit="[nA]",
                                                    plotSize=(800,600),
                                                    plotPercent::Tuple{AbstractFloat,AbstractFloat}=(0.0,1.0),
                                                    tickfontsize=12,
                                                    xlabelfontsize=12,
                                                    ylabelfontsize=12,
                                                    plotConductances=false,
                                                    trueData=false,
                                                    offsetTrue=false,
                                                    overlay=true,
                                                    linewidth=1.5,
                                                    predictionColor=palette(:default)[1],
                                                    kwargs...)
    
    # Recover voltage plots and data to compute currents and conductances
    plt_neurons,t,V̂,X̂,V,Iapp,T,inds = getVoltagePlots(net,iodata;neurons=[n,],tticks=(isnothing(tbar) ? tticks : false),tlabel="",predictionColor=predictionColor, linewidth=linewidth, overlay=overlay, plotPercent=plotPercent,kwargs...)

    # Recover current names
    currentNames = keys(net[n,m].ionicCurrents)
    
    # Recover ionic currents and conductances
    g = conductances(net,V̂,X̂)
    iIon = ionicCurrents(net,V̂,X̂)

    # Plot conductances or ionic currents
    pltCurrents=[]
    for i=1:length(currentNames)
        currentName = currentNames[i]
        if !isnothing(g[n,m][i]) && plotConductances
            y = g[n,m][i][inds]
            str="g"
            unit=gUnit
            dec=2
        else
            y=iIon[n,m][i,inds]
            str="I"
            unit=IUnit
            dec=1
        end
        plt = plot()
        offset=0.0
        if trueData
            ytrue = T
            plot!(plt,t[inds]/1000,ytrue[i,inds],
                    color=:black,
                    opacity=0.5,
                    style=:dash,
                    linewidth=linewidth)
            offsetTrue ? offset = mean(ytrue[n,inds])-mean(y) : nothing
        end
        plt = plot!(plt,t[inds]/1000,y.+offset,
                    color=predictionColor,
                    linewidth=linewidth,
                    ylabel=string(L"${%$str}_{\mathrm{%$currentName}}$ %$unit"),
                    xticks=(isnothing(tbar) ? tticks : false))
        yticks!(plt,[yticks(plt)[1][1][1],yticks(plt)[1][1][end-1]],
                    [string(round(yticks(plt)[1][1][1],digits=dec)),
                        string(round(yticks(plt)[1][1][end-1],digits=dec))])
        push!(pltCurrents,plt)
    end
    
    if isnothing(tbar)
        plot!(pltCurrents[end],xticks=tticks,xlabel=tlabel)
        bmargin=0.0Plots.mm
    else
        plot!(pltCurrents[end],[t[inds[1]]/1000,t[inds[1]]/1000+tbar],ylims(pltCurrents[end])[1]*[1,1],color=:black,linewidth=3,label=false)
        annotate!(pltCurrents[end],t[inds[1]]/1000+tbar/2, ylims(pltCurrents[end])[1],text("$tbar s",:top,Plots.default(:fontfamily),12))
        bmargin=5.0Plots.mm
    end

    if overlay
        pltVoltage = [plt_neurons[1][:pltV̂],]
        l = (1+length(pltCurrents),1)
    else
        pltVoltage = [plt_neurons[1][:pltV],plt_neurons[1][:pltV̂]]
        l = (2+length(pltCurrents),1)
    end
    plt = plot(pltVoltage...,pltCurrents...,layout=l,size=plotSize,legend=legend,tickfontsize=tickfontsize,xlabelfontsize=xlabelfontsize,ylabelfontsize=ylabelfontsize,
                    bottom_margins=permutedims([[0.0Plots.mm for i=1:(length(pltVoltage)+length(pltCurrents)-1)];bmargin]),
                    left_margin=5.0Plots.mm)
    display(plt)
    return plt
end

function plotCurrents(valDict::Dict,hpList,valData::IOData;plotSize=(1200,900),metric=:AngSep,kwargs...)
    netType = Symbol("net" * string(metric))
    plts = []
    eachPlotSize = plotSize./length(hpList)
    for hp in hpList
        plt = plotCurrents(valDict[hp][netType],valData;
                            title=string(hp),
                            plotSize=eachPlotSize,
                            kwargs...)
        push!(plts,plt)
    end
    if length(plts)==1
        plot(plts[1],size=plotSize)    
    else
        iseven(length(plts)) ? nothing : push!(plts,plot())
        plot(plts...,layout=(round(Int,length(plts)/2),2),size=plotSize)
    end
end

"""
    Plotting functions for convenience
"""
function plotForcedCurrents(net::Network,iodata::IOData;i=1,j=1)
    currentNames = keys(net[n,m].ionicCurrents)
    d=SSData(net,iodata)
    V,X=d.V,d.X
    iLeak = leakCurrents(net,V)
    iIon = ionicCurrents(net,V,X)
    pltV=plot(V[i][:],title="Voltage")
    pltLeak=plot(iLeak[i,i][:],title="Leak current")
    pltIon=[plot(iIon[i,j][k,:],title=string("Total current (",i,",",j,"), ", currentNames[k])) for k=1:length(net[i,j][:ionicCurrents])]
    plot(pltV,pltLeak,pltIon...,layout=(length(pltIon)+2,1),size=(800,800))
end

function plotForcedConductances(net::Network,iodata::IOData;i=1)
    currentNames = keys(net[n,m].ionicCurrents)
    d=SSData(net,iodata)
    V,X=d.V,d.X
    g = conductances(net,V,X)
    pltV=plot(V[i][:],title="Voltage")
    plt=[(!isnothing(g[i,i][j]) ? plot(g[i,i][j][1,:],title=string(currentNames[j], " conductance")) : plot()) for j=1:length(g[i,i])]
    plot(pltV,plt...,layout=(length(plt)+1,1),size=(800,800))
end

function plotForcedActivations(net::Network,iodata::IOData;i=1,k=1)
    currentNames = keys(net[n,m].ionicCurrents)
    d=SSData(net,iodata)
    V,X=d.V,d.X
    y = activations(net,V,X)
    pltV=plot(V[i][:],title="Voltage")
    plts=[]
    for j=1:length(y[i,i])
        plt=plot()
        if !isnothing(y[i,i][j][1]) 
            plt = plot!(plt,y[i,i][j][1][k,:],title=currentNames[j],label="activation")
        end
        if !isnothing(y[i,i][j][2])
            plt = plot!(plt,y[i,i][j][2][k,:],title=currentNames[j],label="inactivation")
        end
        if !isnothing(y[i,i][j][1]) || !isnothing(y[i,i][j][2])
            push!(plts,plt)
        end
    end
    plot(pltV,plts...,layout=(length(plts)+1,1),size=(800,800))
end

"""
    Function used to plot Multiple shooting resutls shot by shot
    Currently only working if MSData was constructed with a single IOData
"""
function multiple_shooting_plot(net::Network,d::MSData,shotIndex::Tuple{Int64, Int64};neuron_index=1,plotStates=false,tUnit=:index)
    Ni,Nf = shotIndex
    V̂seq,X̂seq = net(d)
    V̂,X̂ = shotTrajectories(V̂seq,X̂seq)
    
    tinds = [(i-1)*d.shotsize:i*d.shotsize-1 for i = Ni:Nf]
    binds = [tinds[i][1] for i=1:(Nf-Ni+1)]
    if tUnit == :index 
        dt = 1 
        tLabel = "k [samples]"
    elseif tUnit == :ms
        dt = net.cell.dt
        tLabel = "t [ms]"
    else
        error("tUnit must be :index or :ms")
    end

    # Plot validation data
    plt=plot(reduce(vcat,tinds)*dt,d.rawdata[1].V[neuron_index][reduce(vcat,tinds).+ 1],xlabel=tLabel)
    if plotStates
        pltX=[plot(xlabel=tLabel) for j=1:size(X̂[1][neuron_index][neuron_index],1)]
    end
    
    # Plot batch trajectories
    for (i,n) in enumerate(Ni:Nf)
        plt = plot!(plt,tinds[i]*dt,vec(V̂[n][neuron_index]),color=:orange,legend=false)
        if plotStates
            for j=1:size(X̂[1][neuron_index][neuron_index],1)
                pltX[j] = plot!(pltX[j],tinds[i]*dt,X̂[n][neuron_index][neuron_index][j,:],color=:orange,legend=false)
            end
        end
    end
    
    # Plot initial conditions
    plt=scatter!(plt,binds*dt,d.V₀[neuron_index].value[1,Ni:Nf],color=:black,markersize = 2,legend=false)
    if plotStates
        for j=1:size(X̂[1][neuron_index][neuron_index],1)
            pltX[j]=scatter!(pltX[j],binds*dt,d.X₀[neuron_index,neuron_index].value[j,Ni:Nf],color=:black,markersize = 2,legend=false)
        end
        return plt,pltX
    else
        return plt
    end
end

function create_spike_train(voltage_data, threshold; offset=false)
    N = length(voltage_data)
    translation = 0.0
    if offset==true
        translation = threshold
    end
    spike_train = translation*ones(N)
    for i in 2:N-1
        if voltage_data[i] > threshold && voltage_data[i] > voltage_data[i-1] && voltage_data[i] > voltage_data[i+1]
            spike_train[i] = translation+1
        end
    end
    return spike_train
end

# Function to smooth spike trains with a Gaussian kernel
function smooth_spike_train(spike_train, rho)
    # Calculate the size of the Gaussian kernel
    kernel_size = ceil(Int, 100 * rho)
    # Create the Gaussian kernel
    gaussian_kernel = exp.(-((1:kernel_size) .- (kernel_size / 2)).^2 / (2 * rho^2))
    # gaussian_kernel ./= sum(gaussian_kernel)  # Normalize the kernel
    gaussian_kernel ./= sqrt(2*pi*rho^2)  # Normalize the kernel with 1/
    # Perform the full convolution
    conv_result = DSP.conv(spike_train, gaussian_kernel)
    # Calculate the start and end points to extract the 'same' part of the convolution
    pad_size = floor(Int, kernel_size / 2)
    return conv_result[pad_size+1:end-pad_size]
end

# Function to calculate the angular separation between smoothed spike trains
function angular_separation(smoothed_train1, smoothed_train2)
    dot_product = dot(smoothed_train1, smoothed_train2)
    norm1 = norm(smoothed_train1)
    norm2 = norm(smoothed_train2)
    # return dot_product / (norm1 * norm2)
    return dot_product / max(norm1^2,norm2^2)
end

"""
    Plots validation data for the trained ANN model
"""
function ANNplot(net::NetworkCell,data,inds_plot;overlay=true)
    t₀ = 1
    plt = Vector{Any}(undef,length(inds_plot))
    for m = inds_plot
        d = data.data[2][m]
        x = t₀:length(d)
        y = d[t₀:end]
        ŷ = fv̇(net,data.data[1]...)[m][t₀:end]
        println("MSE: ",Flux.mse(y,ŷ))
        if overlay
            plt[m] = plot(x,y,label="measured dv/dt(t)",linecolor="orange", xlabel = "k [# samples]")
            plt[m] = plot!(x,ŷ,linecolor="blue",label="predicted dv̂/dt(t)")
        else
            plt1 = plot(x,y,label="measured dv/dt(t)",linecolor="orange", xlabel = "k [# samples]")
            plt2 = plot(x,ŷ,linecolor="blue",label="predicted dv̂/dt(t)")
            plt[m] = plot(plt1,plt2,layout=(2,1))
        end
    end
    return plt
end

# Add temperature later
function ANNplotChannel(net,V,X;lbl) #,clr
    t = (1:length(V[1]))*net.cell.dt
    plt_i = []
    for i = 1:size(X,1)
        plt_j = []
        for j = 1:size(X,2)
            y = net.cell.ANN[i][:ionicCurrent][j]((V[i],X[i,j],nothing))
            ttl=string("Ionic/synaptic current from neuron ",j," to neuron ",i,".")
            plt = plot(t,vec(y),xlabel="t [ms]",title=ttl, label=lbl[i,j]) #color=clr[i,j],
            push!(plt_j,plt)
        end
        push!(plt_i,plt_j)
    end
    return plt_i
end

function ANNplotChannel(net,data,neuron_inds,channel_inds;overlay=false,lw=1)
    neuron_plt = []
    for n = neuron_inds
        d = data[n].data
        channel_plt = []
        for m = channel_inds
            x = 1:length(d[1][1])
            ŷ = net.ANN[n,n][:ionicCurrent][m](d[1][3][m])

            voltage_plt = plot(x,vec(vec(d[1][3][m][1])),linecolor="blue",label="v(t)",lw=lw)
            output_plt = plot(x,vec(ŷ),linecolor="blue",label="ANN output",lw=lw)

            if overlay==true
                state_plt = plot(x,vec(d[1][3][m][2][1,:]),lw=lw)
                for k = 2:length(net.FB[n,m].state0)
                    state_plt = plot!(x,vec(d[1][3][m][2][k,:]),lw=lw)
                end
            else
                state_plt = []
                for k = 1:length(net.FB[n,m].state0)
                    push!(state_plt,plot(x,vec(d[1][3][m][2][k,:]),lw=lw))
                end
                state_plt = plot(state_plt...,layout=(length(state_plt),1))
            end
            push!(channel_plt,(voltage=voltage_plt,output=output_plt,states=state_plt))
        end
        push!(neuron_plt,channel_plt)
    end
    return neuron_plt
end