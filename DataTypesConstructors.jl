# Quick fix: modifies the inputs so that iapp has dynamics.
# TO DO: incorporate this in model, with dynamical input states and so on.
function IOData(τ_input, ltiType, d::IOData)
    inputFB = [ltiType(τ_input,d.dt,trainable=false) for i=1:length(d.I)]
    Ivec = [vcat(d.I[i],warmUp(inputFB[i],d.I[i][:])) for i=1:length(d.I)]
    I = Tuple(Ivec)
    return IOData(d.V,I,d.T,d.t,d.dt)
end

function IOData(τ_input, ltiType, d::Vector{D}) where D<:IOData
    return [IOData(τ_input,ltiType,d[i]) for i=1:length(d)]
end

# Takes single input-output dataset wrt neuron and creates input-output dataset wrt the ann
function ANNData(FB::Tuple, d::IOData, dumpTransientPercent::AbstractFloat)
    # Get rid of transients
    t₀ = getInitTimeIndex(FB,d,dumpTransientPercent)
    V = [V[:,t₀:end-1] for V in d.V]
    I = [(isnothing(I) ? fill(nothing,1,length(V[1])) : I[:,t₀:end-1]) for I in d.I]
    T = isnothing(d.T) ? fill(nothing,1,length(V[1])) : d.T[:,t₀:end-1]
    dV = [(V[:,t₀+1:end]-V[:,t₀:end-1])/d.dt for V in d.V]

    # Get internal states and get rid of transients
    X = Matrix{Matrix{eltype(V[1])}}(undef,length(FB),length(FB[1]))
    # Must consider sampling factor
    f = samplingFactor(d.dt,FB[1,1].dt)
    for i = 1:length(FB)
        for j = 1:length(FB[i])         
            f > 1 ? Vj = repeat(d.V[j],inner=(1,f)) : Vj = d.V[j]
            X[i,j] = warmUp(FB[i,j],Vj[:])
            X[i,j] = X[i,j][:,t₀*f:f:end-1]
        end
    end
    return ANNData(V,X,I,T,dV)
end

# Takes vector of input-output datasets wrt neuron and creates input-output dataset wrt the ann
function ANNData(FB::Tuple, data::Vector{D}, dumpTransientPercent::AbstractFloat) where D<:IOData
    annData = ANNData(FB,data[1],dumpTransientPercent)
    V,X,I,T,dV = annData.V,annData.X,annData.I,annData.T,annData.dV
    for d in data[2:end]
        annData₊ = ANNData(FB,d,dumpTransientPercent)
        V = [hcat(V[i],annData₊.V[i]) for i in 1:length(V)]
        I = [hcat(I[i],annData₊.I[i]) for i in 1:length(I)]
        T = hcat(T,annData₊.T)
        dV = [hcat(dV[i],annData₊.dV[i]) for i in 1:length(dV)]
        X = [hcat(X[i,j],annData₊.X[i,j]) for i in 1:size(X,1), j in 1:size(X,2)]
    end
    return ANNData(V,X,I,T,dV)
end

# Takes input-output dataset wrt the ann and organizes data for training the ANN
function setup_ANNData(data::ANNData, xpu; batchsize = Inf, shuffle=true, partial=false, rng=Random.GLOBAL_RNG)
    # Construct tuples according to the channel ANN structures
    input = (Tuple(data.V),Tuple([Tuple([data.X[i,j] for j=1:size(data.X,2)]) for i=1:size(data.X,1)]),Tuple(data.I))
    output = Tuple(data.dV)
    input,output = map(xpu,(input,output))

    # Use dataloader to create minibatches
    batchsize > length(data.V[1]) ? batchsize = length(data.V[1]) : nothing
    annData = Flux.DataLoader((input,output); batchsize=batchsize, shuffle=shuffle, partial=partial, rng=rng)
    return annData
end

# Takes input-output dataset wrt neuron and organizes data for training the ANN
function setup_ANNData(netcell::NetworkCell, iodata::D, xpu; σ=0.0, batchsize = Inf, shuffle=true, partial=false, rng=Random.GLOBAL_RNG) where D<:Union{IOData,Vector}
    # Create non-batched data
    data = ANNData(netcell.FB,iodata,netcell.transientPercent)
    if σ>0.0
        data = addNoise(data,σ,rng)
    end
    return setup_ANNData(data,xpu,batchsize=batchsize,shuffle=shuffle,partial=partial,rng=rng)
end

# Adds noise to the data so that the SNR is σ
function addNoise(data::ANNData,σ::AbstractFloat,rng=Random.GLOBAL_RNG)
    σ=Float32(σ)
    X = [X+σ*std(X,dims=2).*randn(rng,Float32,size(X)) for X in data.X]
    # V = [V+σ*std(V,dims=2).*randn(rng,Float32,size(V)) for V in data.V]
    # I = [(eltype(I)==Nothing ? I : I+σ*std(I,dims=2).*randn(rng,Float32,size(I))) for I in data.I]
    # T = eltype(data.T)==Nothing ? T : data.T+σ*std(data.T,dims=2).*randn(rng,Float32,size(data.T))
    # return ANNData(V,X,I,T,data.dV)
    return ANNData(data.V,X,data.I,data.T,data.dV)
end

"""
    Takes a single i/o data trial and puts the data in a single batch multiple-shooting format.
""" 
function MSData(net::Network,d::IOData; dumpTransients=true, 
                                        train_ic = true, 
                                        shotsize::Int=typemax(Int))
    
    local firstshot::Int,lastshot::Int,n₀::Int,N::Int,initindex::Int
    t₀ = dumpTransients ? getInitTimeIndex(net.cell,d) : 1      # initial discrete-time index in full dataset
    T = length(d.V[1])                                          # total number of points in full dataset
    
    lastshot = floor(Int,T/shotsize)                            # index of final shot
    if lastshot <= 1                                            # only one shot possible   
        firstshot = 1                                           # index of initial shot
        lastshot = 1                                            # correct index of final shot
        n₀ = t₀                                                 # initial discrete-time index in single shot
        N = T                                                   # final discrete-time index in single shot
        shotsize = N - n₀ + 1
    else                                                        # multiple shots possible
        firstshot = ceil(Int,t₀/shotsize)                       # correct index of first shot
        firstshot > lastshot ? firstshot = lastshot : nothing   # not enough full shots after dumping, use last shot only
        n₀ = 1                                                  # initial discrete-time index in each shot
        N = shotsize                                            # final discrete-time index in each shot
    end

    # Create multiple shots of data
    Vseq = Tuple(Tuple(reduce(hcat,[d.V[i][:,(s-1)*shotsize+n] for s=firstshot:lastshot]) for i=1:net.size[1]) for n=n₀:N)
    Iseq = Tuple(Tuple((isnothing(d.I[i]) ? nothing : reduce(hcat,[d.I[i][:,(s-1)*shotsize+n] for s=firstshot:lastshot])) for i=1:net.size[1]) for n=n₀:N)

    # If some of the voltage traces are inputs (instead of states of models to be estimated)
    if net.size[1]<net.size[2]
        Useq = Tuple(Tuple(reduce(hcat,[d.V[i][:,(s-1)*shotsize+n] for s=firstshot:lastshot]) for i=net.size[1]+1:net.size[2]) for n=n₀:N)
    else
        Useq = Tuple(nothing for n=n₀:N)
    end

    if isnothing(d.T)
        Tseq = Tuple(nothing for n=n₀:N)
    else
        Tseq = Tuple(reduce(hcat,[d.T[:,(s-1)*shotsize+n] for s=firstshot:lastshot]) for n=n₀:N)
    end

    # Warm up the initial internal states
    V₀ = deepcopy(Vseq[1])
    X₀ = Matrix{eltype(d.V)}(undef,net.size)
    f = samplingFactor(d.dt,net.cell.dt)
    for i=1:net.size[1]
        for j=1:net.size[2]
            Vj = f > 1 ? repeat(d.V[j],inner=(1,f)) : d.V[j]            
            Xsim = warmUp(net.cell.FB[i,j],Vj[:])
            X₀[i,j] = reduce(hcat,[Xsim[:,(s-1)*shotsize*f+n₀] for s=firstshot:lastshot])
        end
    end

    # Set initial conditions
    V₀ = Tuple(InitialCondition(V₀[i]) for i=1:length(V₀))
    X₀ = Tuple(Tuple(InitialCondition(X₀[i,j]) for j=1:size(X₀,2)) for i=1:size(X₀,1))

    # Raw data from where shots were taken
    initindex = (firstshot-1)*shotsize+n₀
    rawV = Tuple(d.V[i][:,initindex:end] for i = 1:net.size[2])
    rawI = Tuple((isnothing(d.I[i]) ? nothing : d.I[i][:,initindex:end]) for i = 1:net.size[1])
    isnothing(d.T) ? rawT = nothing : rawT = d.T[:,initindex:end]
    rawt = d.t[initindex:end]
    rawdata = (IOData(rawV,rawI,rawT,rawt,d.dt),)
    
    return MSData(Vseq,Useq,Iseq,Tseq,V₀,X₀,shotsize,(1+lastshot-firstshot,),rawdata,train_ic,d.dt,f)
end

"""
    Takes multiple i/o data trials and puts the data in a single batch multiple-shooting format.
    This might need to revision for efficiency.
""" 
function MSData(net::Network,iodata::Vector{D}; dumpTransients=true, train_ic = true, shotsize = Inf) where D<:IOData
    data = [MSData(net,iodata[i],dumpTransients=dumpTransients, train_ic = train_ic, shotsize = shotsize) for i=1:length(iodata)]

    # In case the requested shotsize exceeds the smallest dataset length
    shotsize_new = minimum([d.shotsize for d in data])
    if shotsize_new != shotsize
        println("Warning: shotsize was reduced to ",shotsize_new," for all datasets. Some data may not be used.")
        shotsize = shotsize_new
    end

    Vseq = Tuple([Tuple([hcat([data[d].Vseq[k][i] for d=1:length(data)]...) for i=1:net.size[1]]) for k=1:shotsize])
    Iseq = Tuple([Tuple([hcat([data[d].Iseq[k][i] for d=1:length(data)]...) for i=1:net.size[1]]) for k=1:shotsize])
    
    if net.size[1] < net.size[2]
        Useq = Tuple([Tuple([hcat([data[d].Useq[k][i] for d=1:length(data)]...) for i=1:net.size[2]-net.size[1]]) for k=1:shotsize])
    else
        Useq = Tuple([nothing for k=1:shotsize])
    end
    if isnothing(iodata[1].T)
        Tseq = Tuple([nothing for k=1:shotsize])
    else
        Tseq = Tuple([hcat([data[d].Tseq[k] for d=1:length(data)]...) for k=1:shotsize])
    end

    V₀ = Tuple([InitialCondition(hcat([data[d].V₀[i].value for d=1:length(data)]...)) for i=1:net.size[1]])
    X₀ = Tuple([Tuple([InitialCondition(hcat([data[d].X₀[i,j].value for d=1:length(data)]...)) for j=1:net.size[2]]) for i=1:net.size[1]])
    nshots = Tuple([data[d].nshots[1] for d =1:length(data)])
    rawdata = Tuple([data[d].rawdata[1] for d=1:length(data)])
    return MSData(Vseq,Useq,Iseq,Tseq,V₀,X₀,shotsize,nshots,rawdata,train_ic,data[1].dt,data[1].samplingFactor)
end

"""
    Takes multiple i/o data trials and puts the data in a multiple batch multiple-shooting format,
    where each batch corresponds to one i/o data trial.
""" 
function MSBatches(net::N,data::Vector{D}; dumpTransients=true, train_ic = true, shotsize = Inf, rng=Random.GLOBAL_RNG) where {N<:Network,D<:IOData}
    batches = [MSData(net,data[i],dumpTransients=dumpTransients, train_ic = train_ic, shotsize = shotsize) for i=1:length(data)]
    return RNNBatches(batches,[i for i=1:length(batches)],length(batches),rng)
end

"""
    Takes multiple i/o data trials and puts the data in a multiple mini-batch multiple-shooting format,
    with mini-batches containing a specific number of shots.
""" 
function MSMiniBatches(net::N,data::Vector{D}; shotsPerBatch::Int, shotsize = 10, dumpTransients=true, train_ic = true, partial::Bool=false, rng=Random.GLOBAL_RNG) where {N<:Network,D<:IOData}
    trialData = [MSData(net,data[i],dumpTransients=dumpTransients, train_ic =train_ic, shotsize=shotsize) for i=1:length(data)]
    trialMiniBatches = [MSMiniBatches(trialData[i],shotsPerBatch=shotsPerBatch,partial=partial,rng=rng) for i=1:length(trialData)]
    allMiniBatches = vcat([trialMiniBatches[i].batches for i=1:length(trialMiniBatches)]...)
    return RNNBatches(allMiniBatches,rng=rng)
end

"""
   Creates multiple-shooting batches from data in multiple shooting format.
   WARNING: if the multiple shooting dataset is a merged dataset from different trials, this will lead 
   to wrong results since some mini-batches will mix up data from different trials.
"""
function MSMiniBatches(d::MSData;shotsPerBatch::Int,partial::Bool=true,rng=Random.GLOBAL_RNG)
    if shotsPerBatch < 1
        throw(ErrorException("Invalid number of shots per batch."))
    else
        batches = MSData[]
        nfullbatches = floor(Int,d.nshots[1]/shotsPerBatch)
        shots_in_last_batch = d.nshots[1] - nfullbatches*shotsPerBatch
        
        println("Shots in the last batch: ",shots_in_last_batch)
        local incomplete_batch
        if shots_in_last_batch > 0 && partial
            incomplete_batch = 1
        else
            incomplete_batch = 0
        end

        for b = 1:nfullbatches+incomplete_batch
            local nshots    
            if b <= nfullbatches
                nshots = shotsPerBatch
            else
                nshots = shots_in_last_batch
            end
            Vseq = Tuple([Tuple([d.Vseq[n][i][:,(1:nshots).+(b-1)*shotsPerBatch] for i=1:length(d.Vseq[n])]) for n=1:d.shotsize])
            Useq = Tuple([(isnothing(d.Useq[n]) ? nothing : Tuple([d.Useq[n][i][:,(1:nshots).+(b-1)*shotsPerBatch] for i=1:length(d.Useq[n])])) for n=1:d.shotsize])
            Iseq = Tuple([Tuple([d.Iseq[n][i][:,(1:nshots).+(b-1)*shotsPerBatch] for i=1:length(d.Iseq[n])]) for n=1:d.shotsize])
            Tseq = Tuple([(isnothing(d.Tseq[n]) ? nothing : d.Tseq[n][:,(1:nshots).+(b-1)*shotsPerBatch]) for n=1:d.shotsize])
            V₀ = Tuple([InitialCondition(d.V₀[i].value[:,(1:nshots).+(b-1)*shotsPerBatch]) for i=1:length(d.V₀)])
            X₀ = Tuple([Tuple([InitialCondition(d.X₀[i,j].value[:,(1:nshots).+(b-1)*shotsPerBatch]) for j=1:length(d.X₀[i])]) for i=1:length(d.X₀)])

            # Raw data from where shots were taken
            t₀ = (b-1)*shotsPerBatch*d.shotsize+1
            fin = (b-1)*shotsPerBatch*d.shotsize+nshots*d.shotsize
            rawV = Tuple([d.rawdata[1].V[i][:,t₀:fin] for i = 1:length(d.rawdata[1].V)])
            rawI = Tuple([d.rawdata[1].I[i][:,t₀:fin] for i = 1:length(d.rawdata[1].I)])
            rawT = isnothing(d.rawdata[1].T) ? nothing : d.rawdata[1].T[:,t₀:fin]
            rawt = d.rawdata[1].t[t₀:fin]
            rawdata = IOData(rawV,rawI,rawT,rawt,d.dt)

            batch = MSData(Vseq,Useq,Iseq,Tseq,V₀,X₀,d.shotsize,(nshots,),(rawdata,),d.train_ic,d.dt,d.samplingFactor)
            push!(batches,batch)
        end
        return RNNBatches(batches,[i for i=1:length(batches)],length(batches),rng)
    end
end

"""
    Constructor used to set up the data for closed-loop training of the network model 
        using backwards-euler-type iterations
""" 
function SSData(net::Network,d::IOData; dumpTransients=true)
    t₀ = dumpTransients ? getInitTimeIndex(net.cell,d) : 1           # initial discrete-time index in full dataset
    N = length(d.V[1])                                              # total number of points in full dataset
        
    V = Tuple([d.V[i][:,t₀:end] for i = 1:net.size[1]])
    # X = [M(net.cell.FB[i,j](d.V[j][:])[:,t₀:end]) for i=1:net.size[1], j=1:net.size[2]]
    X = Tuple([Tuple([warmUp(net.cell.FB[i,j],d.V[j][:])[:,t₀:end] for j=1:net.size[2]]) for i=1:net.size[1]])
    U = net.size[1] == net.size[2] ? nothing : Tuple([d.V[i][:,t₀:end] for i=net.size[1]+1:net.size[2]])
    I = Tuple([(isnothing(d.I[i]) ? nothing : d.I[i][:,t₀:end]) for i = 1:net.size[1]])
    T = isnothing(d.T) ? nothing : T = d.T[:,t₀:end]

    return SSData(V,X,U,I,T,1+N-t₀,Float32(d.dt))
end

"""
    Creates backwards-euler batches from a single backwards-euler dataset 
"""
function SSBatches(d::SSData, batchsize::Int; rng=Random.GLOBAL_RNG)
    if batchsize > d.length                             # not enough data for a full batch                         
        nbatches = 1                                    # correct index of final shot
        batchsize = d.length
    else                                                # at least one full batch possible
        nbatches = floor(Int,d.length/batchsize)            # correct index of final shot
    end

    # Network size
    m = size(d.X,1)
    n = size(d.X,2)

    # Create multiple batches of data
    Vseq = [[d.V[i][:,(s-1)*batchsize+1:s*batchsize] for i=1:m] for s=1:nbatches]
    Xseq = [[d.X[i,j][:,(s-1)*batchsize+1:s*batchsize] for i=1:m, j=1:n] for s=1:nbatches]
    Iseq = [[(isnothing(d.I[i]) ? nothing : d.I[i][:,(s-1)*batchsize+1:s*batchsize]) for i=1:m] for s=1:nbatches]

    if isnothing(d.U)
        Useq = [nothing for s=1:nbatches]
    else
        Useq = [[d.V[i][:,(s-1)*batchsize+1:s*batchsize] for i=m+1:n] for s=1:nbatches]
    end

    if isnothing(d.T)
        Tseq = [nothing for s=1:nbatches]
    else
        Tseq = [d.T[:,(s-1)*batchsize+1:s*batchsize] for s=1:nbatches]
    end

    batches = [SSData(Vseq[s],Xseq[s],Useq[s],Iseq[s],Tseq[s],batchsize) for s=1:nbatches]

    return RNNBatches(batches,[i for i=1:nbatches],nbatches,rng)
end

# Downsamples the MSData with a new batchsize that is a multiple of the previous batchsize: (new batch size) = (old batch size)*mult
# TO DO : REVISE DV
function downsample(d::MSData,mult::Int)
    lastshot = floor(Int,d.nshots/mult)*mult

    Vseq = [[d.Vseq[n][i][:,1:mult:lastshot] for i=1:length(d.Vseq[n])] for n=1:d.shotsize]
    Iseq = [[d.Iseq[n][i][:,1:mult:lastshot] for i=1:length(d.Iseq[n])] for n=1:d.shotsize]
    if isnothing(d.Useq)
        Useq=nothing
    else
        Useq = [[d.Useq[n][i][:,1:mult:lastshot] for i=1:length(d.Useq[n])] for n=1:d.shotsize]
    end
    
    # dVseq = []
    for m=1:(mult-1)
        Vseq₊ = [[d.Vseq[n][i][:,(1+m):mult:lastshot] for i=1:length(d.Vseq[n])] for n=1:d.shotsize]
        Iseq₊ = [[d.Iseq[n][i][:,(1+m):mult:lastshot] for i=1:length(d.Iseq[n])] for n=1:d.shotsize]
        # dVseq₊ = [[d.dVseq[n][i][:,(1+m):mult:lastshot] for i=1:length(d.dVseq[n])] for n=1:d.shotsize-1]
        append!(Vseq,Vseq₊)
        append!(Iseq,Iseq₊)
        # append!(dVseq,dVseq₊)
        if !isnothing(d.Useq)
            Useq₊ = [[d.Useq[n][i][:,(1+m):mult:lastshot] for i=1:length(d.Useq[n])] for n=1:d.shotsize]
            append!(Useq,Useq₊)
        end
    end
    V₀ = [d.V₀[i].value[:,1:mult:lastshot] for i = 1:length(d.V₀)]
    X₀ = [d.X₀[i,j].value[:,1:mult:lastshot] for i=1:size(d.X₀,1), j=1:size(d.X₀,2)]
    V₀ = [InitialCondition(V₀[i]) for i=1:length(V₀)]
    X₀ = [InitialCondition(X₀[i,j]) for i=1:size(X₀,1), j=1:size(X₀,2)]
    
    return MSData(Vseq,Useq,Iseq,Vseq,V₀,X₀,d.shotsize*mult,floor(Int,d.nshots[1]/mult),d.rawdata,d.train_ic,d.dt,d.samplingFactor)
end