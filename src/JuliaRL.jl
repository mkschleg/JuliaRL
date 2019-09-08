
module JuliaRL

using Random
using Logging

greet() = println("Hello Reinforcement Learning!")


export
    AbstractEnvironment,
    get_actions,
    get_reward,
    is_terminal,
    AbstractAgent,
    start!,
    step!

include("core/environment.jl")
include("core/agent.jl")

export
    GVF,
    Horde,
    GVFParamFuncs
include("core/gvf.jl")

export
    AbstractFeatureConstructors,
    create_features,
    feature_size
include("core/feature_constructors.jl")

export
    TileCoder
include("features/TileCoder.jl")

export
    RandomAgent
include("agent/random.jl")


end
