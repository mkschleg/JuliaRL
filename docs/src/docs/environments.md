
```@meta
CurrentModule = JuliaRL
```

```@docs
AbstractEnvironment
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

