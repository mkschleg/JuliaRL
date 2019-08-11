


# Update a normal Flux Model!

function update!(model, opt, loss)
    grads = Flux.gradient(()->loss, params(model))
    for weights in params(model)
        Flux.Optimise.update!(opt, weights, grads[weights])
    end
end



mutable struct SingleLayer{F, FP, A}
    σ::F
    σ′::FP
    W::A
end

SingleLayer(in::Integer, out::Integer, σ, σ′; init=(dims...)->zeros(Float32, dims...)) =
    SingleLayer(σ, σ′, init(out, in))

(layer::SingleLayer)(x) = layer.σ.(layer.W*x)
deriv(layer::SingleLayer, x) = layer.σ′.(layer.W*x)

# Sparse Updates

Linear(in::Integer, out::Integer) =
    SingleLayer(in, out, identity, (x)->1.0)

sigmoid′(x) = sigmoid(x)*(1.0-sigmoid(x))
identity′(x) = 1.0
tanh′(x) = 1 - (tanh(x))^2

mutable struct TabularModel{A}
    W::A
end
Tabular(dims::Integer...) = TabularModel(zeros(dims...))

@forward TabularModel.W Base.getindex, Base.setindex!

(layer::TabularModel)(x) = layer[x]
(layer::TabularModel)(x::Array{T, 1}) where {T<:Integer} = layer[x...]

mutable struct SparseModel{F, FP, A, P}
    σ::F
    σ′::FP
    W::A
    ϕ::P
end

SparseModel(num_weights::Integer, out::Integer, σ=identity, σ′=identity′; init=(dims...)->zeros(Float32, dims...)) =
    SparseModel(σ, σ′, init(out, num_weights), zeros(Float32, out, num_weights))

(layer::SparseModel)(x::Array{Int64, 1}) = layer.σ.(sum(layer.W[x]))
deriv(layer::SparseModel, x::Array{Int64, 1}) = layer.σ′.(sum(layer.W[x]))

(layer::SparseModel)(x::Array{CartesianIndex{1}, 1}) = layer.σ.(sum(layer.W[x]))
deriv(layer::SparseModel, x::Array{CartesianIndex{1}, 1}) = layer.σ′.(sum(layer.W[x]))
