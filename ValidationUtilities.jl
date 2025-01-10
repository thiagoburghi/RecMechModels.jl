# Must change to allow validation with multiple datasets
function validate(snapshots,valdata::IOData; fc_low::Real=1/10,
                                             fc_high::Real=1/2,
                                             threshold::Real=-Inf,
                                             std::Real=50.0,
                                             prefilter::Bool=true)    
    # Some admin
    filter = digitalfilter(Bandpass(fc_low, fc_high, fs=1/valdata.dt), Butterworth(6))
    cutoff = round(Int, 6.0 * std)
    k = cutoff:-1:-cutoff
    gaussian_kernel = exp.(-k.^2 / (2 * std^2))
    gaussian_kernel /= sum(gaussian_kernel)

    # Validation data
    # COMPUTE THIS IN THE DISTRIBUTED LOOP IN CASE WE LEARN TIMESCALES
    RNNvaldata = MSData(snapshots[1][:model],valdata)
    n_neurons = snapshots[1][:model].size[1]

    V = [RNNvaldata.rawdata[1].V[n][:] for n=1:n_neurons]
    V_f = similar(V)
    for n=1:n_neurons
        # Bandpass filtering to isolate spike frequency    
        if prefilter==true
            v_f = filtfilt(filter, V[n])   
        else
            v_f = V[n]
        end
        # Threshold to isolate spike timings
        threshold > -Inf ? v_f = relu.(v_f .- threshold) : nothing
        # Smooth 
        if std>0.0
            v_f = DSP.conv(v_f,gaussian_kernel) # old v_f,inds_f = filter_signal(L,v_f)
            v_f = v_f[cutoff+1:end-cutoff]
        end
        V_f[n] = v_f
    end
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
        V̂_f = similar(V̂)
        for n=1:n_neurons
            # Bandpass filtering to isolate the spikes
            if prefilter==true
                v̂_f = filtfilt(filter,V̂[n])   
            else
                v̂_f = V̂[n]
            end
            # Threshold to isolate spike timings
            threshold > -Inf ? v̂_f = relu.(v̂_f .- threshold) : nothing
            # Smooth 
            if std>0.0
                v̂_f = DSP.conv(v̂_f,gaussian_kernel) # old v̂_f,_ = filter_signal(L,v̂_f)
                v̂_f = v̂_f[cutoff+1:end-cutoff]
            end
            V̂_f[n] = v̂_f
            predicted_traces[i][:unfiltered][n][:] = V̂[n]
            predicted_traces[i][:filtered][n][:] = V̂_f[n]
        end

        # Compute closed-loop MSE
        cl_loss[i] = sum([Flux.mse(V̂_f[n],V_f[n]) for n=1:n_neurons])/n_neurons
        ang_sep[i] = sum([angular_separation(V̂_f[n],V_f[n]) for n=1:n_neurons])/n_neurons 
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
            netLastEpoch=snapshots[end][:model],
            data=data_traces,valLoss=cl_loss,angSep=ang_sep,trainLoss=ol_loss,epochs=epochs,times=times)
end

function validate_conductances(snapshots,valdata::IOData,datapath,filename)
    dt = valdata.dt
    m_hh,h_hh,n_hh = load_HH(string(datapath,filename),proptrain);
    g_Na_hh = 120*m_hh.^3 .* h_hh;
    g_K_hh = 36*n_hh.^4;

    RNNvaldata = MSData(snapshots[1][:model],valdata)
    t  = RNNvaldata.rawdata[1].t

    g_Na_hh = g_Na_hh[(round(Int,t[1]/dt)+1):end]/maximum(g_Na_hh[(round(Int,t[1]/dt)+1):end])
    g_K_hh = g_K_hh[(round(Int,t[1]/dt)+1):end]/maximum(g_K_hh[(round(Int,t[1]/dt)+1):end])

    data_traces = (unfiltered=g_Na_hh,filtered=g_K_hh)
    epochs = SharedArray{Int}(length(snapshots))
    ol_loss = SharedArray{Float64}(length(snapshots))
    cl_loss = SharedArray{Float64}(length(snapshots))
    predicted_traces = @DArray [deepcopy(data_traces) for _=1:length(snapshots)]

    @sync @distributed for i=1:length(snapshots)
        net = snapshots[i][:model]

        # Simulate model
        V̂seq,X̂seq = net(RNNvaldata)
        V̂,X̂ = shotTrajectories(V̂seq,X̂seq)   
        G,maxG = conductances(net,V̂[1],X̂[1])
        ĝ = G[1,1]
        ĝ_Na = ĝ[1,:]/maximum(ĝ[1,:])
        ĝ_K = ĝ[2,:]/maximum(ĝ[2,:])
        
        # Compute closed-loop MSE
        cl_loss[i] = Flux.mse(ĝ_Na,g_Na_hh)
        cl_loss[i] += Flux.mse(ĝ_K,g_K_hh)
        epochs[i] = snapshots[i][:epoch]
        ol_loss[i] = snapshots[i][:loss]
        predicted_traces[i][1][:] = ĝ_Na
        predicted_traces[i][2][:] = ĝ_K
        println(string("Simulated model number ",i," of ",length(snapshots)))
    end
    cl_loss[isnan.(cl_loss)] .= Inf
    return cl_loss,ol_loss,epochs,data_traces,collect(predicted_traces)
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
                                        hull=false,
                                        metric=:AngSep,
                                        plotBest=true,
                                        deltaEpochIndex=10,
                                        initEpochIndex=1,
                                        opacityAfterFirst=0.3,
                                        legendfontsize=12,
                                        linewidth=2.0,
                                        legend=true)
    plts = plotBest ? [plot() for _ in 1:4] : [plot() for _ in 1:2]
    local bestHP
    opct = 1.0
    for hp in sort(collect(keys(valDict)))
        # Do we want to filter / hull the data?
        if hull
            window=[1,1]/2
            valLoss = convexHull(valDict[hp][:valLoss]) #DSP.conv(valDict[hp][:valLoss],window)[2:end-2] #
            angSep = concaveHull(valDict[hp][:angSep])  #DSP.conv(valDict[hp][:angSep],window)[2:end-2]#
            epochs = valDict[hp][:epochs]               # valDict[hp][:epochs][2:end-1]
        else
            valLoss = valDict[hp][:valLoss]
            angSep = valDict[hp][:angSep]
            epochs = valDict[hp][:epochs]
        end

        # Get times to print in x tick labels
        tTicks = (epochs[initEpochIndex:deltaEpochIndex:end],[string("Epoch ",epochs[i]," \n", getTime(valDict[hp][:times][i])) for i=initEpochIndex:deltaEpochIndex:length(epochs)])

        # Training loss plot
        plot!(plts[1],epochs,log10.(valDict[hp][:trainLoss]);
                                    title=title,
                                    ylabel=string(lossType,"\n log training loss"),
                                    label=string(hpName,hp),
                                    legend=legend,
                                    xticks=tTicks,
                                    legendfontsize=legendfontsize,
                                    linewidth=linewidth,
                                    opacity=opct)
        # Validation plots
        if metric == :AngSep
            # Traces
            plot!(plts[2],epochs,angSep;
                            ylabel=string(hull ? "max " : "","validation \n angular separation"),
                            ylims=(0,1),
                            label=string(hpName,hp),
                            xticks=tTicks,
                            linewidth=linewidth,
                            legend=false,
                            opacity=opct)
            bestHP = sort(collect(keys(valDict)), by = hp -> valDict[hp][:angSep][valDict[hp][:maxAngSepInd]], rev=true)
            if plotBest
                best_angsep = [valDict[hp][:angSep][valDict[hp][:maxAngSepInd]] for hp in sort(collect(keys(valDict)))]  
                best_times = [valDict[hp][:times][valDict[hp][:maxAngSepInd]] for hp in sort(collect(keys(valDict)))]
                # Best results
                plot!(plts[3],sort(collect(keys(valDict))),best_angsep;
                                                title="Best model angular separation",
                                                xlabel=hpName,
                                                marker=:circle,
                                                legend=false,
                                                linewidth=linewidth,
                                                # yticks=(best_angsep,[round(a,digits=2) for a in best_angsep])
                                                )
                # Times
                plot!(plts[4],sort(collect(keys(valDict))),best_times;
                                                title="Best model training time",
                                                xlabel=hpName,
                                                legend=true,
                                                linewidth=linewidth,
                                                marker=:circle,
                                                yticks=(best_times,[getTime(t) for t in best_times]))
            end
        elseif metric == :ValLoss
            # Traces
            plot!(plts[2],epochs,log10.(valLoss),
                                            ylabel=string(lossType,"\n log validation loss"),
                                            label=string(hpName,hp),
                                            legend=false,
                                            xticks=tTicks,
                                            linewidth=linewidth,
                                            opacity=opct)
            bestHP = sort(collect(keys(valDict)), by = hp -> valDict[hp][:valLoss][valDict[hp][:minValLossInd]], rev=false)
            if plotBest
                best_losses = [valDict[hp][:valLoss][valDict[hp][:minValLossInd]] for hp in sort(collect(keys(valDict)))]
                best_times = [valDict[hp][:times][valDict[hp][:minValLossInd]] for hp in sort(collect(keys(valDict)))]
                # Best results
                plot!(plts[3],sort(collect(keys(valDict))),best_losses,
                                                title="Best model validation loss",
                                                xlabel=hpName,
                                                marker=:circle,
                                                legend=false,
                                                linewidth=linewidth,
                                                # yticks=(best_losses,[log10(l) for l in best_losses])
                                                )

                # Times
                plot!(plts[4],sort(collect(keys(valDict))),best_times;
                                                title="Best model training time",
                                                xlabel=hpName,
                                                legend=true,
                                                linewidth=linewidth,
                                                marker=:circle,
                                                yticks=(best_times,[getTime(t) for t in best_times]))
            end
            opct = opacityAfterFirst
        else
            error("Metric not recognized. Possible values are :AngSep and :ValLoss")        
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
        println([getTime(valDict[hp][:times][valDict[hp][:maxAngSepInd]]) for hp in bestHP])
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
        println([getTime(valDict[hp][:times][valDict[hp][:minValLossInd]]) for hp in bestHP])
    end

    plt = plot(plts...,layout=(length(plts),1),size=plotSize,leftmargin=5Plots.mm,bottommargin=5Plots.mm)
    display(plt)
    return bestHP,plts
end

function getTime(t::Float64)
    t = round(Int,t)
    h = floor(Int,t/3600)
    m = floor(Int,(t-h*3600)/60)
    s = t-h*3600-m*60
    return string(h,"h ",m,"m ",s,"s")
end

function plotValVoltages(val::NamedTuple; 
                    title::String="Title",
                    plotPercent::Tuple{AbstractFloat,AbstractFloat}=(0.0,1.0),
                    vlim=:auto,
                    plotSize=(600,400),
                    filtered=false,
                    xlabel="time [ms]",
                    vLabel=true,
                    overlap=true,
                    vLegend=true,
                    vTicks=([-50,-30],["-50mV","-30mV"]),
                    legend=true,
                    metric=:AngSep,
                    linewidth=1.0)

    n_neurons=length(val[:data][1])
    L = length(val[:data][1][1])
    inds = floor(Int,L*plotPercent[1])+1:floor(Int,L*plotPercent[2])
    L = length(inds)
    netType = Symbol("net" * string(metric))
    predictionMetric = Symbol("prediction" * string(metric))
    t = (1:L)*val[netType].cell.dt/1000

    plts=[]
    for n=1:n_neurons
        plt1=plot(t,val[:data][1][n][inds],
            label = vLegend ? string("target") : false,
            ylim=vlim,
            yticks=vTicks,
            ylabel=(isa(vLabel,Bool) ? (vLabel ? string("v",n) : "") : vLabel[n]),
            xlabel="",
            xformatter = (x -> ""),
            title=(n==1 ? title : ""),
            legend=(n==n_neurons ? legend : false),
            linewidth=linewidth)
        plt2 = overlap ? plt1 : plot()
        plt2 = plot!(plt2,t,val[predictionMetric][1][n][inds],
            label = vLegend ? string("prediction") : false,
            ylim=vlim,
            yticks=vTicks,
            ylabel=(isa(vLabel,Bool) ? (vLabel ? string("v",n) : "") : vLabel[n]),
            xlabel=(n==n_neurons ? xlabel : ""),
            xformatter=(n==n_neurons ? (x -> x) : (x -> "")),
            color=palette(:default)[2],
            legend=(n==n_neurons ? legend : false),
            linewidth=linewidth)
        overlap ? push!(plts,plt2) : append!(plts,[plt1,plt2])
        if filtered
            plt3=plot(t,val[:data][2][n][inds],label=string("Data ",n),ylabel="Filtered traces")
            push!(plts,plot(plt3,t,val[predictionMetric][2][n][inds],label=string("Filtered ",n)))
        end
    end
    plt = plot(plts..., leftmargin=5Plots.mm,
                        bottommargin=hcat([-7.5Plots.mm for i=1:1, j_=1:length(plts)-1],[5Plots.mm;;]),
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
function plotIV(valDict::Dict;  n = 1,
                                m = 1,
                                hyParsName::Union{Nothing,String}=nothing,
                                vrange=(-65,-20),
                                plotSize=(800,600),
                                metric=:AngSep)
    # Some admin
    V̄ = vrange[1]:0.01:vrange[2]
    netType = Symbol("net" * string(metric))
    
    # Recover current names
    currentNames = keys(valDict[first(valDict)[1]][netType][n,m].ionicCurrents)
    n_currents = length(currentNames)

    isnothing(hyParsName) ? hyParsName = "Hyperparameter" : nothing

    # Initialize plots
    plt_tot,plt_leak,plt_ion = plot(),plot(), [plot() for _ in 1:n_currents]

    # Plot IV curves for the network associated to each hyperparameter
    for hp in sort(collect(keys(valDict)))
        # Recover model
        net = valDict[hp][netType]
    
        # Compute IV curves
        IVion,IVleak=IV(net,V̄,V̄)
        plotIV!(plt_tot,plt_leak,plt_ion,V̄,IVleak[n],IVion[n,m],currentNames,lb="$hyParsName = $hp")
    end
    iseven(n_currents) ? nothing : push!(plt_ion,plot())
    plot(plt_tot,plt_leak,plt_ion...,
        layout=(2,ceil(Int,(n_currents+2)/2)), 
        size=plotSize,
        margin=5.0Plots.mm)
end

function plotIV!(plt_tot,plt_leak,plt_ion,V̄,IVleak,IVion,currentNames;lb="label")
    IVtot = sum(IVion,dims=1)[:] + IVleak[:]
    # Plot Leak curves
    plt_leak=plot!(plt_leak,V̄[:],IVleak[:],
                                legend=false,
                                xlabel="v [mV]",
                                ylabel="[nA]",
                                title="Leak IV curve",
                                label=lb)
    # Plot Ion ionicCurrents
    for i=1:length(currentNames)
        plt_ion[i]=plot!(plt_ion[i],V̄[:],IVion[i,:],
                                legend=false,
                                xlabel="v [mV]",
                                ylabel="[nA]",
                                title=string(currentNames[i]," current IV curve"),
                                label=lb)
    end
    # Plot Total current
    plt_tot=plot!(plt_tot,V̄[:],IVtot[:],legend=:bottomright,
                                xlabel="v [mV]",
                                ylabel="[nA]",
                                title="Steady-state IV curve",
                                label=lb)
end

"""
    Function used to plot prediction vs data for a model
"""
function getVoltagePlots(net::Network,data::IOData; neurons::AbstractVector{Int}=[1,],
                                                    plotPercent::Tuple{AbstractFloat,AbstractFloat}=(0.0,1.0),
                                                    vlims=:auto,
                                                    vticks=:auto,
                                                    Iticks=:auto,
                                                    tticks=:auto,
                                                    tlabel="t [ms]",
                                                    overlay=false,
                                                    linewidth=1.5,
                                                    predictionColor=palette(:default)[1],
                                                    gtf=false)
    # Simulate the network
    if !gtf 
        t,V̂,X̂,V,Iapp,T = net(data)
    else
        t,V̂,X̂,V,Iapp,T = teacher(net,data)
    end

    # Get the indices to plot
    L = length(t)
    inds = floor(Int,L*plotPercent[1])+1:floor(Int,L*plotPercent[2])

    # Plot the predictions and the voltage
    plt_neurons = []
    for n in neurons
        plt1 = plot(t[inds]/1000,V[n][inds],xlabel=tlabel,xticks=tticks,yticks=vticks,ylims=vlims,color=:black,linewidth=linewidth,ylabel=L"v \;\; \mathrm{(target)}")        
        plt2 = plot(t[inds]/1000,V̂[n][inds],xlabel="",xticks=tticks,yticks=vticks,ylims=vlims,linewidth=linewidth,color=predictionColor,ylabel=L"v \;\; \mathrm{(prediction)}")
        if overlay
            plt2 = plot!(t[inds]/1000,V[n][inds],xlabel=tlabel,color=:black,linewidth=linewidth,opacity=0.5,style=:dash,ylabel="")
        end
        plt3 = plot(t[inds]/1000,Iapp[n][inds],xticks=tticks,yticks=Iticks,color=:black,ylabel=L"$I_{\mathrm{app}}$")
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
    plot(plts...,layout=(1,length(plts)),framestyle=:grid,legend=legend,labelfontsize=labelfontsize,tickfontsize=tickfontsize,leftmargin=5Plots.mm)
end

function plotConductances(net::Network,iodata::IOData; kwargs...)
    plotCurrents(net,iodata; plotConductances=true, kwargs...)
end

function plotCurrents(net::Network,iodata::IOData;  n=1,    #neuron number
                                                    m=1,    #total current number
                                                    currentTicks=:auto,
                                                    legend=false,
                                                    tticks=:auto,
                                                    tlabel="t [s]",
                                                    plotSize=(800,600),
                                                    plotPercent::Tuple{AbstractFloat,AbstractFloat}=(0.0,1.0),
                                                    tickfontsize=12,
                                                    xlabelfontsize=12,
                                                    ylabelfontsize=12,
                                                    plotConductances=false,
                                                    conductanceData=false,
                                                    overlay=true,
                                                    linewidth=1.5,
                                                    predictionColor=palette(:default)[1],
                                                    kwargs...)
    
    # Recover voltage plots and data to compute currents and conductances
    plt_neurons,t,V̂,X̂,V,Iapp,T,inds = getVoltagePlots(net,iodata; tlabel=vtlabel="",predictionColor=predictionColor, linewidth=linewidth, overlay=overlay, plotPercent=plotPercent, tticks=false, kwargs...)

    # Recover current names
    currentNames = keys(net[n,m].ionicCurrents)
    n_currents = length(currentNames)

    # Get current ticks 
    if currentTicks==:auto
        currentTicks = [:auto for _ in 1:n_currents]
    else
        if length(currentTicks) != n_currents
            error("Number of current yticks must match the number of ionicCurrents.")
        end
    end
    
    # Recover ionic currents or conductances
    if plotConductances
        g = conductances(net,V̂,X̂)
        iIon = ionicCurrents(net,V̂,X̂)
        pltCurrents=[]
        for i=1:length(g[n,m])
            currentName=currentNames[i]
            if isnothing(g[n,m][i])
                y=iIon[n,m][i,inds]
                str="I"
            else
                y = g[n,m][i][inds]
                str="g"
            end
            plt = plot(t[inds]/1000,y,
                        color=predictionColor,
                        linewidth=linewidth,
                        # ylabel=string(L"\mathrm{%$currentName \;\; %$str}"),
                        ylabel=string(L"{%$str}_{\mathrm{%$currentName}}"),
                        yticks=currentTicks[i],
                        xticks=(i==length(g[n,m]) ? tticks : false),
                        xlabel=(i==length(g[n,m]) ? tlabel : ""))
                if conductanceData
                    gtrue = T
                    plot!(plt,t[inds]/1000,gtrue[i,inds],
                            color=:black,
                            opacity=0.5,
                            style=:dash,
                            linewidth=linewidth)
                end
            push!(pltCurrents,plt)
        end
    else
        iIon = ionicCurrents(net,V̂,X̂)
        pltCurrents = [plot(t[inds]/1000,iIon[n,m][i,inds],
            linewidth=linewidth,
            color=predictionColor,
            ylabel=string(L"\mathrm{%$currentName current}"),
            yticks=currentTicks[i],
            xticks=(i==size(iIon[n,m],1) ? tticks : false),
            xlabel=(i==length(g[n,m]) ? tlabel : ""),
            leftmargin=5Plots.mm,
            ) for i=1:size(iIon[n,m],1)]
    end
    
    if overlay
        pltVoltage = [plt_neurons[n][:pltV̂],]
        l = (1+length(pltCurrents),1)
    else
        pltVoltage = [plt_neurons[n][:pltV],plt_neurons[n][:pltV̂]]
        l = (2+length(pltCurrents),1)
    end
    plt = plot(pltVoltage...,pltCurrents...,layout=l,size=plotSize,legend=legend,framestyle=:grid,tickfontsize=tickfontsize,xlabelfontsize=xlabelfontsize,ylabelfontsize=ylabelfontsize,
                    bottom_margins=permutedims([0.0Plots.mm for i=1:(length(pltVoltage)+length(pltCurrents))]),
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
function multiple_shooting_plot(net::Network,d::MSData,shotIndex::Tuple{Int64, Int64};neuron_index=1,plotStates=false)
    Ni,Nf = shotIndex
    V̂seq,X̂seq = net(d)
    V̂,X̂ = shotTrajectories(V̂seq,X̂seq)
    
    tinds = [(i-1)*d.shotsize:i*d.shotsize-1 for i = Ni:Nf]
    binds = [tinds[i][1] for i=1:(Nf-Ni+1)]
    
    # Plot validation data
    plt=plot(reduce(vcat,tinds),d.rawdata[1].V[neuron_index][reduce(vcat,tinds).+ 1])
    if plotStates
        pltX=[plot() for j=1:size(X̂[1][neuron_index][neuron_index],1)]
    end
    
    # Plot batch trajectories
    for (i,n) in enumerate(Ni:Nf)
        plt = plot!(plt,tinds[i],vec(V̂[n][neuron_index]),color=:orange,legend=false)
        if plotStates
            for j=1:size(X̂[1][neuron_index][neuron_index],1)
                pltX[j] = plot!(pltX[j],tinds[i],X̂[n][neuron_index][neuron_index][j,:],color=:orange,legend=false)
            end
        end
    end
    
    # Plot initial conditions
    plt=scatter!(plt,binds,d.V₀[neuron_index].value[1,Ni:Nf],color=:black,markersize = 2,legend=false)
    if plotStates
        for j=1:size(X̂[1][neuron_index][neuron_index],1)
            pltX[j]=scatter!(pltX[j],binds,d.X₀[neuron_index,neuron_index].value[j,Ni:Nf],color=:black,markersize = 2,legend=false)
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
    return dot_product / (norm1 * norm2)
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

function convexHull(x::AbstractVector)
    return [minimum(x[1:i]) for i=1:length(x)]
end

function concaveHull(x::AbstractVector)
    x = [x[i]==-Inf ? 0.0 : x[i] for i=1:length(x)]
    return [maximum(x[1:i]) for i=1:length(x)]
end