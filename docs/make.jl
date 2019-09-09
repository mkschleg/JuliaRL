push!(LOAD_PATH,"../src/")

using Documenter, JuliaRL

makedocs(
    sitename="JuliaRL",
    modules = [JuliaRL],
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Home"=>"index.md",
        "Manual" => Any[
             "Environments" => "manual/environment.md"
             # "Agents" => "docs/agents.md"
             # "Learning" => "docs/learning.md"
             # "Feature Creators" => "docs/feature_creators.md"
             ],
         "Documentation" => Any[
             "Environments" => "docs/environments.md"
             "Agents" => "docs/agents.md"
             "GVF" => "docs/gvf.md"
             # "Learning" => "docs/learning.md"
             "Feature Constructor" => "docs/feature_creators.md"
             ]
    ]
)

deploydocs(
    repo = "github.com/mkschleg/JuliaRL.jl.git",
    devbranch = "master"
)
