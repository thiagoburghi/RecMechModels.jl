"""
    Open-loop least-squares loss function.
"""
struct Loss{F,R<:Union{Float32,Nothing},G<:Union{AbstractMatrix,Nothing}} <: AbstractLoss
    regfun::F                      # Regularization function
    ρ::R                           # Regularization constant
    g::G                           # Symmetric impulse response of a non-causal filter
end

function Loss()
    return Loss(nothing,nothing,nothing)
end

function Loss(regfun,ρ::AbstractFloat)
    return Loss(regfun,convert(Float32,ρ),nothing)
end

function Loss(regfun,ρ::AbstractFloat,g::AbstractArray)
    g = convert.(Float32,g)
    g = reshape(g,length(g),1)
    return Loss(regfun,convert(Float32,ρ),g)
end

function Loss(regfun,ρ::AbstractFloat,λ::Real,dt::AbstractFloat,L::Int)
    if λ != Inf
        g = zeros(2*L+1)
        g[L+1] = 1.0
        for i = 1:L
            g[L+1-i] = exp(-λ*i*dt)
            g[L+1+i] = exp(-λ*i*dt)
        end
        g = g / sum(g)
        g = convert.(Float32,g)
        g = reshape(g,length(g),1)
        return Loss(regfun,convert(Float32,ρ),g)
    else
        return Loss(regfun,convert(Float32,ρ),nothing)
    end
end

# Non-filtered loss
function (l::Loss{F,R,Nothing})(x::Tuple,y::Tuple,net::NetworkCell) where {F,R}
    ŷ = fv̇(net,x...)
    loss = sum(Flux.mse(ŷ[i],y[i]) for i=1:net.size[1])
    return loss
end

# Filtered loss
function (l::Loss{F,R,G})(x::Tuple,y::Tuple,net::NetworkCell) where {F,R,G<:AbstractMatrix}
    loss=0
    ŷ = fv̇(net,x...)
    for n = 1:net.size[1]
        e = ŷ[n] .- y[n]
        ef = filter_error(l,e)
        loss += sum(ef.^2)/length(ef)
    end
    return loss
end

function filter_error(l::Loss{F,R,G},u::AbstractMatrix) where {F,R,G<:AbstractMatrix}
    L = floor(Int,length(l.g)/2)
    M = l.g * u
    ef = M[end,1:end-2*L]
    for i=1:2*L
        ef += M[end-i,1+i:end-2*L+i]
    end
    return ef
end

function regularize(l::Loss,net::N) where N<:NetworkCell
    return l.ρ*regularizer(net,l.regfun)
end

Flux.@layer Loss trainable=()