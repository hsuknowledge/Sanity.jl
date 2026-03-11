function fit_gene(model::Sanity, g::Integer, tmp::Sanity_tmparrays)
    x, c = model.counts[g, :], model.log_cell_sizes
    n, C, x2 = sum(x), length(x), vvmapreduce(abs2, +, x)
    q, L = tmp.q, eachrow(model.likelihood)[g]
    delta, var_d = tmp.delta_bin, tmp.var_d_bin
    @tasks for i in 1:length(L)
        @local F = Vector{Float64}(undef, C)
        v, d, e = model.prior_var[i], eachcol(delta)[i], eachcol(var_d)[i]
        vna = v * (n + 1)
        @. d = v * x       # cache v * x
        @. F = d + c + log(vna) # temporary for fitfrac
        fitfrac(q) = sum(womega.(F .- q)) - vna
        q[i] = find_zero(fitfrac, model.q0, Order16())
        @. F = womega(F - q[i]) # satisfies constraint, sum(F) = vna
        @. d -= F          # d = v * x - F
        @. e = inv(F + 1)  # cache 1/(F+1)
        @. F = abs2(F)     # cache F^2
        F2 = sum(F)        # cache sum(F^2)
        @. F *= e          # cache F^2/(F+1)
        mdl = vna - sum(F) # Matrix Determinant Lemma factor * vna
        L[i] = (x2 * v - F2 / v - log(mdl/v) + vvmapreduce(log, +, e)) / 2 - (n+1) * q[i]
        @. e *= v * (1 + F / mdl) # error squared, not adjusting for null counts
    end
    L .= exp.(L .- maximum(L))
    L .= L ./ sum(L)
    qg = vvmapreduce(*, +, L, q)
    model.mu[g] = digamma(n+1) - qg
    model.var_mu[g] = trigamma(n+1) + vvmapreduce(*, +, L, q, q) - abs2(qg)
    delta_out, var_d_out = eachrow(model.delta)[g], eachrow(model.var_delta)[g]
    mul!(delta_out, delta, L) # δg = ∑b δL
    @. delta = abs2(delta) # Var(δg) = -δg² + ∑b δ²L + ∑b Var(δ)L in resp. lines
    @. var_d_out = abs2(delta_out)
    mul!(var_d_out, delta, L, 1, -1)
    mul!(var_d_out, var_d, L, 1, 1)
end
