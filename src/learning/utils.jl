

import LinearAlgebra: dot


_square(x::AbstractArray) = x.*x
_prod(x::Array{T, 1}, y::Array{T, 1}) where {T<:Number} = x.*y
