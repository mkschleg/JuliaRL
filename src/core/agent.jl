export AbstractAgent, start!, step!, get_action

abstract type AbstractAgent end

"""
    start!(agent::AbstractAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

Function for restarting the agent.
"""
function start!(agent::AbstractAgent, env_s_tp1, rng::AbstractRNG=Random.GLOBAL_RNG; kwargs...)
    @error "Implement start! function for agent $(typeof(agent))"
end

"""
    step!(agent::AbstractAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

Function to take a step with an agent
"""
function step!(agent::AbstractAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG; kwargs...)
    @error "Implement step! function for agent $(typeof(agent))"
end

"""
    get_action(agent::AbstractAgent, state; rng=Random.GLOBAL_RNG, kwargs...)

Helper function for developing agents. Returns an action given a state.
"""
function get_action(agent::AbstractAgent, state)
    @error "Implement get Action for agent"
end

