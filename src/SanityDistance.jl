function cell_distance!(model::Sanity, dist::Matrix;
                        B = 40, genes = missing, cells = missing)
    G, C = size(model.counts)
    @assert size(dist) == (C, C) "Shape of `dist` should be C x C."
    g = ismissing(genes) ? findall(distance_snr(model) .>= 1) : (1:G)[genes]
    @info "Calculate distance using " * string(length(g)) * " genes."
    vg = gene_variance(model)[g]
    c = ismissing(cells) ? (1:C) : (1:C)[cells]
    rescale(x, v, e = x) = x * v / max(v - e, 1e-6)
    A1 = rescale.(view(model.delta, g, :), vg, view(model.var_delta, g, :))
    A2 = rescale.(view(model.var_delta, g, :), vg)
    A3 = vg .* ((1:B) * 2 / B)' # sample from alpha in 2/B:2/B:2, 0 is skipped
    (x, eta2, alphav) = eachcol.((A1, A2, A3))
    ll(x2, av, e2) = x2 / (av + e2) + log(av + e2)
    d2(x2, av, e2) = (x2 * av / (av + e2) + e2) * av / (av + e2)
    @time @tasks for i in c
        @set scheduler = :greedy
        @local begin
            x2, e2 = (Vector{Float64}(undef, length(vg)) for _ in 1:2)
            LL, D2 = (Vector{Float64}(undef, B) for _ in 1:2)
        end
        for j in 1:i-1 # lower triangular indices
            @. x2 = abs2(x[i] - x[j])
            @. e2 = eta2[i] + eta2[j]
            for k in 1:B
                LL[k] = vvmapreduce(ll, +, x2, alphav[k], e2) / -2
                D2[k] = vvmapreduce(d2, +, x2, alphav[k], e2)
            end
            LL .= exp.(LL .- maximum(LL))
            dist[i, j] = vmapreducethen(*, +, sqrt, LL, D2) / sqrt(sum(LL))
        end
    end
end
