

# module MountainCar

using Random

module MountainCarConst
const vel_limit = (-0.07, 0.07)
const pos_limit = (-1.2, 0.5)
const pos_initial_range = (-0.6, 0.4)
end

mutable struct MountainCar <: AbstractEnvironment
    pos::Float64
    vel::Float64
    actions::AbstractSet
    MountainCar() = new(0.0, 0.0, Set(0:2))
    MountainCar(rng::AbstractRNG) = new((rand(rng)*(MountainCarConst.pos_initial_range[2]
                                                    - MountainCarConst.pos_initial_range[1])
                                         + MountainCarConst.pos_initial_range[1]),
                                        0.0,
                                        Set(0:2))
end

function reset!(env::MountainCar; rng = Random.GLOBAL_RNG, kwargs...)
    # throw("Implement reset! for environment $(typeof(env))")
    env.pos = (rand(rng)*(MountainCarConst.pos_initial_range[2]
                          - MountainCarConst.pos_initial_range[1])
               + MountainCarConst.pos_initial_range[1])
    env.vel = 0
end

get_actions(env::MountainCar) = env.actions
valid_action(env::MountainCar, action) = action in env.actions

function environment_step!(env::MountainCar, action; rng=Random.GLOBAL_RNG, kwargs...)
    @boundscheck valid_action(env, action)
    env.vel = clamp(env.vel + (action - 1)*0.001 - 0.0025*cos(3*env.pos), MountainCarConst.vel_limit...)
    env.pos = clamp(env.pos + env.vel, MountainCarConst.pos_limit...)
end

function get_reward(env::MountainCar) # -> determines if the agent_state is terminal
    if env.pos >= MountainCarConst.pos_limit[2]
        return 0
    end
    return -1
end

function is_terminal(env::MountainCar) # -> determines if the agent_state is terminal
    return env.pos >= MountainCarConst.pos_limit[2]
end

function get_state(env::MountainCar)
    return get_normalized_state(env)
end

function get_normalized_state(env::MountainCar)
    pos_limit = MountainCarConst.pos_limit
    vel_limit = MountainCarConst.vel_limit
    return [(env.pos - pos_limit[1])/(pos_limit[2] - pos_limit[1]), (env.vel - vel_limit[1])/(vel_limit[2] - vel_limit[1])]
end


