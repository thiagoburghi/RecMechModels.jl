function Flux.gpu(data::Vector{D}) where D<:AbstractData
    return [gpu(d) for d in data]
end

function (l::AbstractLoss)(net::N,data::Union{B,Vector{D}}) where {N<:Network,B<:MiniBatches,D<:AbstractData}
    loss = 0
    for d in data
        loss += l(net,d)
    end
    return loss/length(data)
end

"""
    Multiple shooting loss functions
    NOTICE: The term in the loss given by vₖ-v̂ₖ is dt*(v̇ₖ-v̂̇ₖ).
    So that the loss here can be compared to the open-loop loss,
    the relevant term below has been divided by dt.
"""
struct MSLoss{L<:Function,R<:Union{Function,Nothing},T<:Union{AbstractFloat,Nothing},V<:Union{AbstractFloat,Nothing},X<:Union{Tuple,AbstractFloat,Nothing},D<:Union{AbstractFloat,Nothing},G<:Union{AbstractFloat,Nothing}} <: AbstractLoss
    lossFun::L
    regFun::R
    ρ₀::T
    ρᵥ::V
    ρₓ::X
    δᵥ::D
    ρᵧ::G
    teacher::Bool
end

Flux.@layer MSLoss trainable=()

function MSLoss(;ρ₀=nothing,ρᵥ=nothing,ρₓ::Union{AbstractFloat,Nothing}=nothing,δᵥ=nothing,ρᵧ=nothing,lossFun=Flux.mse,regFun=l2norm,teacher=false)
    ρ₀,ρᵥ,ρₓ,δᵥ,ρᵧ = map(x -> x == 0 ? nothing : x, (ρ₀,ρᵥ,ρₓ,δᵥ,ρᵧ))
    return MSLoss(lossFun,regFun,ρ₀,ρᵥ,ρₓ,δᵥ,ρᵧ,teacher)
end

# If the network is passed as argument, then the internal state regularization is normalized by the DC gain
function MSLoss(net::Network;ρ₀=nothing,ρᵥ=nothing,ρₓ::Union{AbstractFloat,Nothing}=nothing,δᵥ=nothing,ρᵧ=nothing,lossFun=Flux.mse,regFun=l2norm,teacher=false)
    ρ₀,ρᵥ,ρₓ,δᵥ,ρᵧ = map(x -> x == 0 ? nothing : x, (ρ₀,ρᵥ,ρₓ,δᵥ,ρᵧ))
    if !isnothing(ρₓ)
        DC = Tuple(Tuple(DCgain(net.cell.FB[i,j]) for j=1:net.size[2]) for i=1:net.size[1])
        ρₓ_tuple = Tuple(Tuple((sqrt(ρₓ)./DC[i][j]) for j=1:net.size[2]) for i=1:net.size[1])
    else
        ρₓ_tuple = nothing
    end
    return MSLoss(lossFun,regFun,ρ₀,ρᵥ,ρₓ_tuple,δᵥ,ρᵧ,teacher)
end

function predictionLoss!(l::MSLoss,net::N,d::D) where {N<:Network,D<:MSData}
    loss = 0
    reset!(net,d)
    for k=1:d.shotsize
        loss += sum(l.lossFun(net.V[i],d.Vseq[k][i]) for i=1:net.size[1])/net.size[1]
        for s=1:d.samplingFactor
            if !l.teacher
                net(d.Useq[k],d.Iseq[k])
            else
                teacher(net,d.Useq[k],d.Iseq[k],d.Vseq[k]) 
            end
        end
    end
    return loss/(d.shotsize*d.dt^2) # so that the loss is comparable to the open-loop loss
end

# There used to be the function below which is now replaced by teacher with gamma = 1
# function inverseLoss!(l::MSLoss,net::N,d::D) where {N<:Network,D<:MSData}

function (l::MSLoss{L,R,Nothing,Nothing,Nothing,Nothing,Nothing})(net::N,d::D) where {L,R,N<:Network,D<:MSData}
    return predictionLoss!(l,net,d)
end

function (l::MSLoss{L,R,F,Nothing,Nothing,Nothing,Nothing})(net::N,d::D) where {L,R,F<:AbstractFloat,N<:Network,D<:MSData}
    loss = predictionLoss!(l,net,d)
    return loss + l.ρ₀*regularizer(net.cell,l.regFun)
end

function (l::MSLoss{L,R,Nothing,F,Nothing,Nothing,Nothing})(net::N,d::D) where {L,R,F<:AbstractFloat,N<:Network,D<:MSData}
    loss = predictionLoss!(l,net,d)
    # Overlapping states must take different datasets/files into account
    final_loss = 0.0
    ini,fin = 0,0
    for s=1:length(d.nshots) 
        fin += d.nshots[s]
        final_loss += l.ρᵥ*sum(l.lossFun(d.V₀[i].value[2+ini:fin],net.V[i][1+ini:fin-1]) for i=1:net.size[1])/net.size[1]
        ini += d.nshots[s]
    end
    loss += final_loss/(length(d.nshots)*d.dt^2)
    return loss
end

function (l::MSLoss{L,R,F,F,Nothing,Nothing,Nothing})(net::N,d::D) where {L,R,F<:AbstractFloat,N<:Network,D<:MSData}
    loss = predictionLoss!(l,net,d)
    # Overlapping states must take different datasets/files into account
    final_loss = 0.0
    ini,fin = 0,0
    for s=1:length(d.nshots) 
        fin += d.nshots[s]
        final_loss += l.ρᵥ*sum(l.lossFun(d.V₀[i].value[2+ini:fin],net.V[i][1+ini:fin-1]) for i=1:net.size[1])/net.size[1]
        ini += d.nshots[s]
    end
    loss += final_loss/(length(d.nshots)*d.dt^2)
    return loss + l.ρ₀*regularizer(net.cell,l.regFun)
end

function regularizeFinalStates(ρₓ::AbstractFloat,lossFun::Function,X₀,X,ini,fin,sz)
    return ρₓ*sum(lossFun(X₀[i,j].value[:,2+ini:fin],X[i,j][:,1+ini:fin-1]) for i=1:sz[1] for j=1:sz[2])/sz[1]
end

function regularizeFinalStates(ρₓ::Tuple,lossFun::Function,X₀,X,ini,fin,sz)
    return sum(lossFun(ρₓ[i][j].*X₀[i,j].value[:,2+ini:fin],ρₓ[i][j].*X[i,j][:,1+ini:fin-1]) for i=1:sz[1] for j=1:sz[2])/sz[1]
end

function (l::MSLoss{L,R,F,F,X,Nothing,Nothing})(net::N,d::D) where {L,R,F<:AbstractFloat,X<:Union{AbstractFloat,Tuple},N<:Network,D<:MSData}
    loss = predictionLoss!(l,net,d)
    # Overlapping states must take different datasets/files into account
    final_loss = 0.0
    ini,fin = 0,0
    for s=1:length(d.nshots) 
        fin += d.nshots[s]
        final_loss += l.ρᵥ*sum(l.lossFun(d.V₀[i].value[2+ini:fin],net.V[i][1+ini:fin-1]) for i=1:net.size[1])/net.size[1]
        final_loss += regularizeFinalStates(l.ρₓ,l.lossFun,d.X₀,net.X,ini,fin,net.size)
        ini += d.nshots[s]
    end
    loss += final_loss/(length(d.nshots)*d.dt^2)
    return loss + l.ρ₀*regularizer(net.cell,l.regFun)
end

function (l::MSLoss{L,R,Nothing,F,X,Nothing,Nothing})(net::N,d::D) where {L,R,F<:AbstractFloat,X<:Union{AbstractFloat,Tuple},N<:Network,D<:MSData}
    loss = predictionLoss!(l,net,d)
    # Overlapping states must take different datasets/files into account
    final_loss = 0.0
    ini,fin = 0,0
    for s=1:length(d.nshots) 
        fin += d.nshots[s]
        final_loss += l.ρᵥ*sum(l.lossFun(d.V₀[i].value[2+ini:fin],net.V[i][1+ini:fin-1]) for i=1:net.size[1])/net.size[1]
        final_loss += regularizeFinalStates(l.ρₓ,l.lossFun,d.X₀,d.X,ini,fin,net.size)
        ini += d.nshots[s]
    end
    loss += final_loss/(length(d.nshots)*d.dt^2)
    return loss
end

# Derivative regularization
function (l::MSLoss{L,R,F,F,<:Union{AbstractFloat,Tuple},F,Nothing})(net::Network,d::MSData) where {L,R,F<:AbstractFloat}
    loss = predictionLoss!(l,net,d)
    # Overlapping states must take different datasets/files into account
    final_loss = 0.0
    ini,fin = 0,0
    for s=1:length(d.nshots) 
        fin += d.nshots[s]
        # Compute initial/final voltage errors
        final_loss += l.ρᵥ*sum(l.lossFun(d.V₀[i].value[2+ini:fin],net.V[i][1+ini:fin-1]) for i=1:net.size[1])/net.size[1]
        
        # Compute initial/final derivative errors
        Itot₀ = [net.cell.ANN[i,j](d.V₀[i].value[:,2+ini:fin],d.X₀[i,j].value[:,2+ini:fin]) for i=1:net.size[1], j=1:net.size[2]]
        Itot = [net.cell.ANN[i,j](net.V[i][:,1+ini:fin-1],net.X[i,j][:,1+ini:fin-1]) for i=1:net.size[1], j=1:net.size[2]]
        final_loss += l.δᵥ*sum(l.lossFun(Itot₀[i,j],Itot[i,j]) for i=1:net.size[1], j=1:net.size[2]) 
        
        # Compute initial/final state errors
        final_loss += regularizeFinalStates(l.ρₓ,l.lossFun,d.X₀,net.X,ini,fin,net.size)
        ini += d.nshots[s]
    end
    loss += final_loss/(length(d.nshots)*d.dt^2)
    return loss + l.ρ₀*regularizer(net.cell,l.regFun)
end