
```@meta
CurrentModule = JuliaRL
```

```@docs
AbstractEnvironment
start!(env::AbstractEnvironment; kwargs...)
start!(env::AbstractEnvironment, start_state; kwargs...)
start!(env::AbstractEnvironment, rng::AbstractRNG; kwargs...)
step!(env::AbstractEnvironment, action; kwargs...)
step!(env::AbstractEnvironment, action, rng::AbstractRNG; kwargs...)
```

The above functions take advantage of the following interface.

```@docs
JuliaRL.reset!
JuliaRL.environment_step!
JuliaRL.get_reward(env::AbstractEnvironment)
JuliaRL.is_terminal(env::AbstractEnvironment)
JuliaRL.get_state(env::AbstractEnvironment)
JuliaRL.get_actions(env::AbstractEnvironment)
```

