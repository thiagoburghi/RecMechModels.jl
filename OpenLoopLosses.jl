"""
    Open-loop least-squares loss function.
"""
struct Loss{R,F}
    regfun::R                      # Regularization function
    ρ::Union{Float32,Nothing}      # Regularization constant
    ϕ::F                           # Extra regularization, if necessary
end

function Loss()
    return Loss(nothing,nothing,nothing)
end

function Loss(regfun,ρ)
    return Loss(regfun,convert(Float32,ρ),nothing)
end

function Loss(regfun,ρ,ϕ)
    return Loss(regfun,convert(Float32,ρ),ϕ)
end

function (l::Loss{R,Nothing})(x::Tuple,y::Tuple,net::NetworkCell) where R
    ŷ = fv̇(net,x...)
    loss = sum(Flux.mse(ŷ[i],y[i]) for i=1:net.size[1])
    return loss
end

function regularize(l::Loss,net::N) where N<:NetworkCell
    return l.ρ*regularizer(net,l.regfun)
end

Flux.@layer Loss trainable=()