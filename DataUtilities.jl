"""
    Loads somatic membrane voltags and aplied ionicCurrents from .mat files. 
    Mandatory arguments:
    ⋅ n is the total number of neurons in the network (membrane voltages in the data)
    ⋅ datapath is the path of the .mat file
    ⋅ propTrainEnd: proportion of the data to stop using as training data (0 to 1)
    
    By default, training data begins at the beginning of the data in the .mat file, and 
    validation data is whaever is left of the data in the .mat file.
    
    Optional arguments:
    ⋅ filt: prefilter data with Butterworth filter of cutoff frequency ωc and order chosen by user
    ⋅ propTrainInit: proportion of the data to begin using as training data (0 to 1)
    ⋅ propVal: proportion of the data to use as validation data; obtained from the end of the data
"""
function loaddata(m::Int,n::Int,datapath::String,dt::AbstractFloat; filt=false, propTrainInit=0.0, propTrainEnd::AbstractFloat, propVal::AbstractFloat, ωc = 1*1e5, order=2)
    # Prefilter data if required
    if filt == true
        datapath = filter_data(m,n,datapath,dt,ωc,order)
    end

    # Read .mat file
    data = matread(datapath)

    # Read all voltages
    data_v = [data[string("V",i)] for i = 1:n]

    # Read ionicCurrents if they exist
    data_iapp = Vector{Union{eltype(data_v),Nothing}}(undef,m)
    for i = 1:m
        if haskey(data,string("I",i))
            data_iapp[i] = data[string("I",i)]
        else
            data_iapp[i] = nothing
        end
    end
    
    # Read temperature if it exists
    if haskey(data,"Temp")
        data_temp = data["Temp"]
    else
        data_temp = nothing
    end

    # Check if options make sense
    if propTrainInit > propTrainEnd
        error("propTrainInit must be ≤ propTrainEnd")
    end

    # Construct training dataset
    t₀ = floor(Int,propTrainInit*length(data_v[1]))+1
    N = floor(Int,propTrainEnd*length(data_v[1]))    # Training data length
    t = Float32.(dt*(t₀-1:N-1))                              
    V = Tuple([Float32.(permutedims(data_v[i][t₀:N])) for i in 1:length(data_v)])
    I = Tuple([(isnothing(data_iapp[i]) ? nothing : Float32.(permutedims(data_iapp[i][t₀:N]))) for i in 1:length(data_iapp)])
    isnothing(data_temp) ? T = nothing : T = Float32.(permutedims(data_temp[t₀:N,:]))
    traindata = IOData(V,I,T,t,Float32(dt))

    # Construct validation dataset
    N̄ = floor(Int,propVal*(length(data_v[1])))    # Validation data length
    t̄ = Float32.(dt*(0:N̄-1))
    V̄ = Tuple([Float32.(permutedims(data_v[i][end-N̄+1:end])) for i in 1:length(data_v)])
    Ī = Tuple([(isnothing(data_iapp[i]) ? nothing : Float32.(permutedims(data_iapp[i][end-N̄+1:end]))) for i in 1:length(data_iapp)])
    isnothing(data_temp) ? T̄ = nothing : T̄ = Float32.(permutedims(data_temp[end-N̄+1:end,:]))
    valdata = IOData(V̄,Ī,T̄,t̄,Float32(dt))

    return [traindata,],[valdata,]
end

function loaddata(m::Int,n::Int,datapaths::Vector{String},dt::AbstractFloat;kwargs...)
    traindata,valdata = loaddata(m,n,datapaths[1],dt;kwargs...)
    for i = 2:length(datapaths)
        traindata₊,valdata₊ = loaddata(m,n,datapaths[i],dt;kwargs...)
        traindata = vcat(traindata,traindata₊)
        valdata = vcat(valdata,valdata₊)
    end
    return traindata,valdata
end

"""
    Filters voltage traces using a butterworth filter.
    ⋅ m is the number of neurons to be estimated
    ⋅ n is the total number of neurons in the network

    TO DO: TEMPERATURE!

"""
function filter_data(m,n,datapath,dt,ωc,order)
    filtdatapath = string(datapath[1:end-4],"-filtered.mat")
    # if !isfile(filtdatapath)
        data = matread(datapath)

        data_v = tuple([data[string("V",i)] for i = 1:n]...)
        data_iapp = tuple([data[string("I",i)] for i = 1:m]...)

        # traindata,valdata = loaddata(m,n,datapath,proptrain,dt; propVal=propVal)
        # return_data = Vector{IOData}(undef, 2) 

        # Sampling frequency
        fs = 1000 * 1/dt;       # in [kHz], since dt is in [ms]
        ts = 1/fs;

        # Create Butterworth low-pass filter
        fc = ωc/(2*pi);           # convert to hertz
        # Wn = pHz/(fs/2);        # normalize by Nyquist freq
        bwfilter = digitalfilter(Lowpass(fc, fs=fs), Butterworth(order))
        
        # Find out filter's transient settling time
        impulse = zeros(500)
        impulse[1] = 1
        g = filt(bwfilter, impulse)
        energy = g .^ 2
        cumulative_energy = cumsum(energy)
        threshold = 0.999 * cumulative_energy[end]
        settling_index = findfirst(cumulative_energy .> threshold)

        # Filter signals and discard filter transients
        ini = 5*settling_index

        for i = 1:n
            V = filtfilt(bwfilter, data_v[i])[ini:end-ini]
            data[string("V",i)] = V
        end

        for i = 1:m
            I =  filtfilt(bwfilter, data_iapp[i])[ini:end-ini]
            data[string("I",i)] = I
        end

        matwrite(filtdatapath,data)
    # end
    return filtdatapath
end

"""
    Plotting functions
"""
# Converts from sequence of spaced samples to (plottable) array of consecutive samples
# Returns a vector V[#batch][#neuron][voltages]
function seq_to_array(Vseq)
    return [[hcat([Vseq[k][i][j] for k = 1:length(Vseq)]) for i=1:length(Vseq[1])] for j=1:length(Vseq[1][1])]
end

function plotSingleNeuron(data::IOData; title="", inds = nothing, plotIapp=false, vlims=:auto, plotSize=(800,600))
    isnothing(inds) ? (inds = 1:length(data.t)) : nothing
    p1=plot(data.t[inds]/1000,data.V[1][inds],title=title,ylabel="Voltage [mV]",ylims=vlims,xlabel="time [s]")
    plts = [p1,]
    if plotIapp
        p2=plot(data.t[inds]/1000,data.I[1][inds],ylabel="Applied current [μA]",xlabel="time [s]")
        push!(plts,p2)
    end
    return plot(plts...,layout=(length(plts),1),size=plotSize,legend=false)
end

function plotSingleNeuron(data::Vector{D}; inds = nothing, plotIapp=false, vlims=(-65,-15), plotSize=(800,600)) where D<:IOData
    plts = [plotSingleNeuron(data[i],inds=inds,plotIapp=plotIapp,vlims=vlims) for i = 1:length(data)]
    for i = 1:length(plts)
        plot!(plts[i],title="PD neuron, file $(i)",legend=false)
    end
    iseven(length(plts)) ? nothing : push!(plts,plot())
    plot(plts...,layout=(ceil(Int,length(data)/2),2),size=plotSize)
end

function plotLPPD(data::IOData; inds = nothing, plotSize=(800,600),plotIapp=false,file="")
    isnothing(inds) ? (inds = 1:length(data.t)) : nothing
    p1=plot(data.t[inds],data.V[2][inds],title=string("PD neuron",file),ylabel="Voltage [mV]")
    p2=plot(data.t[inds],data.V[1][inds],title=string("LP neuron",file),ylabel="Voltage [mV]")
    plts = [p1,p2]
    if plotIapp
        p3=plot(data.t[inds],data.I[1][inds],title="LP applied current",ylabel="Applied urrent [μA]",xlabel="t [ms]")
        push!(plts,p3)
    end
    plot(plts...,layout=(length(plts),1),size=plotSize,legend=false)
end

function plotLPPD(data::Vector{D}; inds = nothing, plotSize=(800,600),plotIapp=false) where D<:IOData
    plts = [plotLPPD(data[i],inds=inds,plotSize=plotSize,plotIapp=plotIapp,file=", file $i") for i = 1:length(data)]
    iseven(length(plts)) ? nothing : push!(plts,plot())
    plot(plts...,layout=(ceil(Int,length(data)/2),2),size=plotSize)
end

function plotHCO(data::IOData; inds = nothing, plotIapp=false, vlims=(-65,-15))
    isnothing(inds) ? (inds = 1:length(data.t)) : nothing
    p1=plot(data.t[inds],data.V[1][inds],title="Neuron 1",ylabel="Voltage [mV]",ylims=vlims)
    p2=plot(data.t[inds],data.V[2][inds],title="Neuron 2",ylabel="Voltage [mV]",ylims=vlims)
    plts = [p1,p2]
    if plotIapp
        p3=plot(data.t[inds],data.I[1][inds],ylabel="Applied current [μA]",xlabel="t [ms]")
        p4=plot(data.t[inds],data.I[2][inds],ylabel="Applied current [μA]",xlabel="t [ms]")
        push!(plts,p3)
        push!(plts,p4)
    end
    return plot(plts...,layout=(length(plts),1),size=(800,600),legend=false)
end

function plotHCO(data::Vector{D}; inds = nothing, plotIapp=false, vlims=(-65,-15),plotSize=(800,600)) where D<:IOData
    plts = [plotHCO(data[i],inds=inds,plotIapp=plotIapp,vlims=vlims) for i = 1:length(data)]
    for i = 1:length(plts)
        plot!(plts[i],legend=false)
    end
    iseven(length(plts)) ? nothing : push!(plts,plot())
    plot(plts...,layout=(ceil(Int,length(data)/2),2),size=plotSize)
end

function filterDict(dict::Dict,constKeys)
    # Filters only keys with the same constant fields
    filteredDict = filter(kv -> all(getfield(kv[1], k) == v for (k, v) in pairs(constKeys)), dict)
    # Finds the varying key field
    allKeys = keys(first(keys(dict)))
    varyingKey = setdiff(allKeys, keys(constKeys))
    if length(varyingKey) != 1
        println("Varying keys: ", varyingKey)
        return Dict(kv[1] => kv[2] for kv in filteredDict)
    else
        println("Varying key: ", varyingKey)
        varyingKey = first(varyingKey)
        # Creates new dict where the varying key field is the key
        return Dict(getfield(kv[1], varyingKey) => kv[2] for kv in filteredDict)
    end
end