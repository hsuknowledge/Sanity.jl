using StatsBase
using Roots: find_zero, Order16, A42
using ReLambertW: womega # Wright omega: womega(x) = lambertw0(exp(x))
using SpecialFunctions: digamma, trigamma
using OhMyThreads: @tasks
using VectorizedReduction: vvmapreduce, vmapreducethen
using LinearAlgebra: mul!

struct Sanity{T<:AbstractMatrix{Float64}}
    counts::T # G x C
    log_cell_sizes::Vector{Float64}
    q0::Float64
    prior_var::Vector{Float64}  # [0.001, 50] in B=160 points
    likelihood::Matrix{Float64} # G x B
    mu::Vector{Float64}
    var_mu::Vector{Float64}
    delta::Matrix{Float64}      # G x C
    var_delta::Matrix{Float64}  # G x C
end

struct Sanity_tmparrays
    q::Vector{Float64}
    delta_bin::Matrix{Float64}
    var_d_bin::Matrix{Float64}
end
Sanity_tmparrays(C::Integer, B::Integer) = Sanity_tmparrays(
    Vector{Float64}(undef, B),
    Matrix{Float64}(undef, C, B),
    Matrix{Float64}(undef, C, B))

include("SanityInference.jl")
include("SanityDistance.jl")

function Sanity(counts::T where T<:AbstractMatrix{Float64};
                vmin = 0.001, vmax = 50, B::Integer = 160, run::Bool = true)
    G, C = size(counts)
    c = vec(log.(sum(counts; dims = 1)))
    q = log(sum(counts)) # log(total) as initial guess for q at any given v
    v = @. vmin * exp(log(vmax / vmin) / (B-1) * (0:B-1))
    m = Sanity(counts, c, q, v, zeros(G, B),
               zeros(G), zeros(G), zeros(G, C), zeros(G, C))
    run || return m
    tmp = Sanity_tmparrays(C, B)
    @time for g in 1:G
        fit_gene(m, g, tmp)
    end
    m
end

gene_variance(model::Sanity) = model.likelihood * model.prior_var
logcounts(model::Sanity) = model.mu .+ model.delta
logcounts_sd(model::Sanity) = model.var_mu .+ model.var_delta
function gene_entropy(model::Sanity)
    fun(p) = entropy(p, length(p))
    map(fun, eachrow(model.likelihood))
end
function distance_snr(model::Sanity)
    fun(d, e) = var(d) / mean(e) # Var(δ) / mean(ϵ²)
    map(fun, eachrow(model.delta), eachrow(model.var_delta))
end
