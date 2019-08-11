
_square(x::AbstractArray) = x.*x
_square(x::Number) = x*x
_prod(x::Array{T, 1}, y::Array{T, 1}) where {T<:Number} = x.*y
_prod(x::Array{T, 1}, y::AbstractArray) where {T<:Number} = x.*y

# tderror(v_t, c, γ_tp1, ṽ_tp1) = v_t .- (c .+ γ_tp1.*ṽ_tp1)

function tderror(ρ_t::Array{Array{T, 1}, 1},
                 v_t::Array{Array{F, 1}, 1},
                 c::Array{Array{T, 1}, 1},
                 γ_tp1::Array{Array{T, 1}, 1},
                 ṽ_tp1::Array{Array{F,1 }, 1},
                 corr_term) where {T <: Number, F <: Number}
    corr_term.*_prod.(ρ_t, (v_t .- (c .+ _prod.(γ_tp1, ṽ_tp1))))
end

tderror(ρ_t, v_t, c, γ_tp1, ṽ_tp1, corr_term) = corr_term.*ρ_t.*(v_t .- (c .+ γ_tp1.*ṽ_tp1))

function tderror(ρ_t::T, v_t::F, c::T, γ_tp1::T, ṽ_tp1::F, corr_term) where {T<:Number, F<:Number}
    corr_term*ρ_t*(v_t - (c + γ_tp1*ṽ_tp1))
end

tdloss(ρ_t::T, v_t::T, c::T, γ_tp1::T, ṽ_tp1::T, corr_term::T) where {T<:AbstractFloat} =
    (1//2)*corr_term*ρ_t*(v_t - (c + γ_tp1*ṽ_tp1))^2

tdloss(ρ_t::T, v_t::F, c::T, γ_tp1::T, ṽ_tp1::Array{T}, corr_term::T) where {T<:AbstractFloat, F<:TrackedArray} =
    ((1//2)*corr_term*ρ_t*(v_t .- (c .+ γ_tp1.*ṽ_tp1)).^2)[1]

function tdloss(ρ_t::Array{Array{T, 1}, 1},
                v_t::AbstractArray,
                c::Array{Array{T, 1}, 1},
                γ_tp1::Array{Array{T,1}, 1},
                ṽ_tp1::Array{Array{T,1}, 1},
                corr_term) where {T<:AbstractFloat}
    target = c .+ _prod.(γ_tp1, ṽ_tp1)
    error = _square.(v_t .- target)
    return (1//2)*sum(sum.(_prod.(ρ_t.*corr_term, error))) * (1//length(ρ_t))
end

function tdloss(ρ_t::Array{T, 1},
                v_t::AbstractArray,
                c::Array{T, 1},
                γ_tp1::Array{T,1},
                ṽ_tp1::Array{T,1},
                corr_term::T) where {T<:AbstractFloat}
    target = c .+ γ_tp1.*ṽ_tp1
    error = _square(v_t .- target)
    return (1//2)*sum(corr_term.*ρ_t.*error)
end
