function cell_distance!(model::Sanity, dist::Matrix;
                        B = 400, genes = missing, cells = missing)
    C = size(model.counts)[2]
    @assert size(dist) == (C, C) "Shape of `dist` should be C x C."
    g = ismissing(genes) ? findall(distance_snr(model) .>= 1) : genes
    @info "Calculate distance using " * string(length(g)) * " genes."
    vg = gene_variance(model)[g]
    rescale(x, v, y = x) = x * v / max(v - y, 1e-6)
    cells = ismissing(cells) ? (1:C) : cells
    A1 = rescale.(model.delta[g, cells], vg, model.var_delta[g, cells])
    A2 = rescale.(model.var_delta[g, cells], vg)
    A3 = vg .* ((1:B) * 2 / B)' # sample from alpha in 2/B:2/B:2, 0 is skipped
    (G, sub_C), (delta, epsilon2, alphav) = size(A1), eachcol.((A1, A2, A3))
    ll(x2, prior, e2) = x2 / (prior + e2) + log(prior + e2)
    d2(x2, prior, e2) = (x2 * prior / (prior + e2) + e2) * prior / (prior + e2)
    @time @tasks for i in cells[1:sub_C]
        @set scheduler = :greedy
        @local begin
            x2, e2 = (Vector{Float64}(undef, G) for _ in 1:2)
            LL, D2 = (Vector{Float64}(undef, B) for _ in 1:2)
        end
        for j in cells[i+1:sub_C]
            @. x2 = abs2(delta[i] - delta[j])
            @. e2 = epsilon2[i] + epsilon2[j]
            for k in 1:B
                LL[k] = vvmapreduce(ll, +, x2, alphav[k], e2) / -2
                D2[k] = vvmapreduce(d2, +, x2, alphav[k], e2)
            end
            LL .= exp.(LL .- maximum(LL))
            dist[j, i] = vmapreducethen(*, +, sqrt, LL, D2) / sqrt(sum(LL))
        end
    end
end
