
module JuliaRL


greet() = println("Hello Reinforcement Learning!")




export start, start!, step, step!

include("core/environment.jl")
include("core/agent.jl")
include("core/gvf.jl")

# include("core.jl")


end
