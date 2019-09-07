"""
    AbstractCumulant
"""
abstract type AbstractCumulant end

function get(cumulant::AbstractCumulant, state_tp1, action_tp1, preds_tp1)
    throw(DomainError("get(CumulantType, args...) not defined!"))
end


"""
    FeatureCumulant

Basic Cumulant which takes the value c_tp1 = s_tp1[idx] for 1<=idx<=length(s_tp1)
"""
struct FeatureCumulant <: AbstractCumulant
    idx::Int
end

get(cumulant::FeatureCumulant, state_tp1, action_tp1, preds_tp1) =
    state_tp1[cumulant.idx]

"""
    PredictionCumulant

Basic cumulant which takes the value c_tp1 = preds_tp1[idx] for 1 \le idx \le length(s_tp1)
"""
struct PredictionCumulant <: AbstractCumulant
    idx::Int
end

get(cumulant::PredictionCumulant, state_tp1, action_tp1, preds_tp1) =
    preds_tp1[cumulant.idx]

"""
    ScaledCumulant

A cumulant which scales another AbstractCumulant
"""
struct ScaledCumulant{F<:Number, T<:AbstractCumulant} <: AbstractCumulant
    scale::F
    cumulant::T
end

get(cumulant::ScaledCumulant, state_tp1, action_tp1, preds_tp1) =
    cumulant.scale*get(cumulant.cumulant, state_tp1, action_tp1, preds_tp1)

"""
    FunctionalCumulant

A cumulant that has a user defined function c_tp1 = f(state_tp1, action_tp1, preds_tp1)
"""
struct FunctionalCumulant{F<:Function}
    f::F
end

get(cumulant::FunctionalCumulant, state_tp1, action_tp1, preds_tp1) =
    cumulant.f(state_tp1, action_tp1, preds_tp1)

