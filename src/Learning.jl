
module Learning

using Flux

include("learning/Layers.jl")
include("learning/Loss.jl")

"""
    LearningUpdate
"""
abstract type LearningUpdate end

update!(model, opt, lu::LearningUpdate, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy; corr_term=1.0) =
    update!(model, opt, lu::LearningUpdate, ρ, s_t, s_tp1, r, γ, terminal)


# Simple Watkins Q-learning

# Currently only works with a model producing a single Q Function!
struct QLearning <: LearningUpdate end

function WatkinsQLoss(q_t, c, γ_tp1, q̃_tp1)
    # TODO Fix to be a bit more general (i.e. multiple q-value functions, or batches)
    target = c + _prod(γ_tp1, maximum.(q̃_tp1))
    error =  _square(q_t - target)
end


function update!(model, opt, lu::QLearning, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, target_policy)

    # setup loss
    # then pass to a function which updates the model based on the handed loss.

    q_t = model(s_t)[a_t]
    q̃_tp1 = Flux.data(model(s_t))
    loss = WatkinsQLoss(q_t, r, γ_tp1, q̃_tp1)

    update!(model, opt, loss)
    
end





end






