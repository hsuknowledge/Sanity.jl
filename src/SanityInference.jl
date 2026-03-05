function fit_gene(model::Sanity, g::Integer, tmp::Sanity_tmparrays)
    x, c = eachrow(model.counts)[g], model.log_cell_sizes
    n, C, x2 = sum(x), length(x), vvmapreduce(abs2, +, x)
    q, L = tmp.q, eachrow(model.likelihood)[g]
    delta, var_d = tmp.delta_bin, tmp.var_d_bin
    @tasks for i in 1:length(L)
        @local y, F = (Vector{Float64}(undef, C) for _ in 1:2)
        v, d = model.prior_var[i], eachcol(delta)[i]
        vna = v * (n+1)
        @. y = v * x + c + log(vna)
        q[i] = find_zero(q -> sum(womega.(y .- q)) - vna, model.q0, Order16())
        @. F = womega(y - q[i]) # F = f * vna where sum(f) = 1
        @. d = v * x - F # d = log(F) - log(vna) - c + q; F + log(F) = y - q
        mdlr = vna - sum(@. abs2(F) / (F + 1)) # Matrix Determinant Lemma * vna
        ldet = vvmapreduce(log1p, +, F) # -C*log(v) cancels out first term in L
        L[i] = (x2 * v - vvmapreduce(abs2, +, F) / v - log(mdlr / vna) - ldet) / 2 - (n+1) * q[i]
        for j in 1:length(x)
            w = v / (F[j] + 1) # identical to the initial upper bound in jmbreda
            if x[j] != 0
                var_d[j, i] = w * (1 + abs2(F[j]) / (F[j] + 1) / mdlr)
                continue
            end
            dL(e) = (e + 2F[j] * (exp(sqrt(e)) - sqrt(e) - 1)) / v - 1
            var_d[j, i] = try find_zero(dL, w / 4, Order16()) catch
                              find_zero(dL, (0, w), A42()) end
        end
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
