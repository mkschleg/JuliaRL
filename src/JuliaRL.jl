
module JuliaRL


greet() = println("Hello Reinforcement Learning!")


export
    AbstractEnvironment,
    AbstractAgent,
    start!,
    step!

include("core/environment.jl")
include("core/agent.jl")
include("core/gvf.jl")

# include("core.jl")


end
