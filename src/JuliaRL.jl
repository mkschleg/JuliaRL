
module JuliaRL

using Random
using Logging

greet() = println("Hello Reinforcement Learning!")


export
    AbstractEnvironment,
    AbstractAgent,
    start!,
    step!

include("core/environment.jl")
include("core/agent.jl")
include("core/gvf.jl")

export
    AbstractFeatureConstructors,
    create_features,
    feature_size
include("core/feature_constructors.jl")

export
    TileCoder
include("features/TileCoder.jl")

export Learning
include("learning/Learning.jl")


end
