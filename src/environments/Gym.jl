using Random
using PyCall
import OpenAIGym

export render, close
import Base.close

mutable struct GymEnv <: AbstractEnvironment
    gymEnv::OpenAIGym.AbstractGymEnv
    # ExampleEnvironment() = new(rand(rng), rand(rng, ExampleEnvConst.state_initial_range), false)
    reward::Float64
    seed::Int64
    function GymEnv(name::Symbol, version::Symbol; rng=nothing, seed=0)
        new_gym = new(OpenAIGym.GymEnv(name, version), 0.0, seed)
        finalizer(close, new_gym)
        return new_gym
    end
end

function reset!(env::GymEnv; rng = nothing, kwargs...)
    OpenAIGym.reset!(env.gymEnv)
    # pycall(env.gymEnv.pyenv[:seed], env.seed)
end

function Base.close(env::GymEnv)
    !ispynull(env.gymEnv.pyenv) && env.gymEnv.pyenv[:close]()
end

get_actions(env::GymEnv) = OpenAIGym.actions(env.gymEnv, nothing)

function environment_step!(env::GymEnv, action; rng = Random.GLOBAL_RNG, kwargs...)
    r, s = OpenAIGym.step!(env.gymEnv, action)
    env.reward = r
end

function get_reward(env::GymEnv) # -> determines if the agent_state is terminal
    return env.reward
end

function get_state(env::GymEnv) # -> get state of agent
    return env.gymEnv.state
end

function is_terminal(env::GymEnv) # -> determines if the agent_state is terminal
    return OpenAIGym.finished(env.gymEnv)
end

function get_normalized_state(env::GymEnv)
    return env.gymEnv.state
end

function Base.show(io::IO, env::GymEnv)
    println(io, get_state(env))
end



render(env::GymEnv, args...; kwargs...) =
    pycall(env.gymEnv.pyenv[:render], PyAny; kwargs...)



