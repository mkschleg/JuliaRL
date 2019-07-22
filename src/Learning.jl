
module Learning

using Flux

include("learning/Layers.jl")

function offpolicy_tdloss(ρ_t, v_t, c, γ_tp1, ṽ_tp1)
    target = c .+ _prod.(γ_tp1, ṽ_tp1)
    error = _square.(v_t .- target)
    return (typeof(ρ_t[1])(0.5))*sum(mean.(_prod.(ρ_t, error))) * (1//length(ρ_t))
end

"""
    LearningUpdate
"""
abstract type LearningUpdate end

update!(model, opt, lu::LearningUpdate, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0) =
    update!(model, opt, lu::LearningUpdate, ρ, s_t, s_tp1, r, γ, terminal)

mutable struct TD <: LearningUpdate end

####
#
# Online version
#
####
function update!(model, opt, lu::TD, ρ::T, s_t, s_tp1, r::T, γ::T, terminal; corr_term=1.0f0) where {T <: Number}
    v_t = model(s_t)
    v_tp1 = model(s_tp1)
    loss = offpolicy_tdloss(ρ*corr_term, v_t, r, γ, Flux.data(v_tp1))
    grads = Flux.gradient(()->loss, params(model))
    for weights in params(model)
        Flux.Optimise.update!(opt, weights, grads[weights])
    end
end


####
# Assume batch
####
function update!(model, opt, lu::TD, ρ::Array{T, 1}, s_t, s_tp1, r::Array{T, 1}, γ::Array{T, 1}, terminal; corr_term=1.0f0) where {T <: Number}
    v_t = model.(s_t)
    v_tp1 = model.(s_tp1)
    loss = offpolicy_tdloss(ρ.*corr_term, v_t, r, γ, Flux.data.(v_tp1))
    grads = Flux.gradient(()->loss, params(model))
    for weights in params(model)
        Flux.Optimise.update!(opt, weights, grads[weights])
    end
end

function update!(model::SingleLayer, opt, lu::TD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    v_t = model(s_t)
    v_tp1 = model(s_tp1)
    dvdt = deriv.(model, s_t)
    δ = ρ.*tderror(v_t, r, γ, v_tp1)
    Δ = mean(δ.*dvdt.*s_t)
    model.W .+= -Flux.Optimise.apply!(opt, model.W, corr_term*Δ)
end

function update!(model::SparseLayer, opt::Descent, lu::TD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    v_t = model(s_t)
    v_tp1 = model(s_tp1)
    dvdt = [deriv(model, s) for s in s_t]
    δ = ρ.*tderror(v_t, r, γ, v_tp1)
    Δ = δ.*dvdt.*1//length(ρ)

    for i in 1:length(ρ)
        model.W[s_t[i]] .-= opt.eta*corr_term*Δ[i]
    end
end

function update!(model::SparseLayer, opt::RMSProp, lu::TD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    v_t = model(s_t)
    v_tp1 = model(s_tp1)

    dvdt = [deriv(model, s) for s in s_t]
    δ = ρ.*tderror(v_t, r, γ, v_tp1)
    fill!(model.ϕ, 0.0f0)
    Δ = δ.*dvdt.*1//length(ρ)
    for i in 1:length(ρ)
        model.ϕ[s_t[i]] .+= corr_term*Δ[i]
    end
    feats = unique(collect(Iterators.flatten(s_t)))

    acc = get!(opt.acc, model.W, zero(model.W))::typeof(model.W)
    acc .*= opt.rho
    acc[feats] .+= (1-opt.rho) .* model.ϕ[feats].^2
    model.W[feats] .-= model.ϕ[feats].*(opt.eta./sqrt.(acc[feats] .+ 1e-8))

end

function update!(model::TabularLayer, opt::Descent, lu::TD, ρ, s_t, s_tp1, r, γ, terminal; corr_term=1.0)
    v_t = model.(s_t)
    v_tp1 = model.(s_tp1)
    δ = ρ.*tderror(v_t, r, γ, v_tp1)
    Δ = corr_term.*δ.*(1.0/length(v_t))
    for (s_idx, s) in enumerate(s_t)
        model.W[s...] -= opt.eta*Δ[s_idx]
    end
end




end






