using Base.Cartesian: @nexprs
using Combinatorics
using KernelAbstractions

macro kbn_sum!(acc, x, c) # Kahan-Babuska-Neumaier method (ref. KahanSummation.jl)
    esc(quote
        t = $acc + $x
        $c -= ($acc - t) + $x # assuming |acc| > |x| && acc is strictly increasing
        $acc = t
    end)
end

@kernel cpu=false inbounds=true function celldistance_256bins!(
        @Const(pairs), @Const(del), @Const(eta), @Const(vg), @Const(alpha), dist_out
    )
    p, a = @index(Global, NTuple) # pair & alpha indices
    ltmp = @localmem Float32 128
    dtmp = @localmem Float32 128
    i1, i2 = pairs[p] >>> 32, pairs[p] & 0xffffffff
    ng = size(vg)[1]
    @private acc_ll1 = 0f0
    @private acc_ll2 = 0f0
    @private acc_dd1 = 0f0
    @private acc_dd2 = 0f0
    cll1 = cll2 = 0f0
    for g in 1:fld(ng, 4)
        @nexprs 4 j->begin
            g_j = 4g - 4 + j
            x2_j = abs2(del[g_j, i1] - del[g_j, i2])
            e2_j = eta[g_j, i1] + eta[g_j, i2]
            av1_j = alpha[2a-1] * vg[g_j]
            av2_j = alpha[2a] * vg[g_j]
            p1_j = inv(av1_j + e2_j)
            p2_j = inv(av2_j + e2_j)
            @kbn_sum!(acc_ll1, x2_j * p1_j, cll1)
            @kbn_sum!(acc_ll2, x2_j * p2_j, cll2)
            acc_dd1 += (x2_j * p1_j * av1_j + e2_j) * av1_j * p1_j
            acc_dd2 += (x2_j * p2_j * av2_j + e2_j) * av2_j * p2_j
        end
        @kbn_sum!(acc_ll1, -log(p1_1 * p1_2 * p1_3 * p1_4), cll1)
        @kbn_sum!(acc_ll2, -log(p2_1 * p2_2 * p2_3 * p2_4), cll2)
    end
    for g in (fld(ng, 4)*4+1):ng
        x2 = abs2(del[g, i1] - del[g, i2])
        e2 = eta[g, i1] + eta[g, i2]
        av1, av2 = alpha[2a-1] * vg[g], alpha[2a] * vg[g]
        p1 = inv(av1 + e2)
        p2 = inv(av2 + e2)
        @kbn_sum!(acc_ll1, x2 * p1, cll1)
        @kbn_sum!(acc_ll2, x2 * p2, cll2)
        acc_dd1 += (x2 * p1 * av1 + e2) * av1 * p1
        acc_dd2 += (x2 * p2 * av2 + e2) * av2 * p2
        @kbn_sum!(acc_ll1, -log(p1), cll1)
        @kbn_sum!(acc_ll2, -log(p2), cll2)
    end
    acc_ll1 = (acc_ll1 - cll1) * -0.5f0
    acc_ll2 = (acc_ll2 - cll2) * -0.5f0
    ltmp[a] = max(acc_ll1, acc_ll2)
    @synchronize()
    if a % 8 == 1
        m1 = max(ltmp[a], ltmp[a + 1])
        m2 = max(ltmp[a + 2], ltmp[a + 3])
        m3 = max(ltmp[a + 4], ltmp[a + 5])
        m4 = max(ltmp[a + 6], ltmp[a + 7])
        ltmp[a] = max(m1, m2, m3, m4)
    end
    @synchronize()
    if a % 64 == 1
        m1 = max(ltmp[a], ltmp[a + 8])
        m2 = max(ltmp[a + 16], ltmp[a + 24])
        m3 = max(ltmp[a + 32], ltmp[a + 40])
        m4 = max(ltmp[a + 48], ltmp[a + 56])
        ltmp[a] = max(m1, m2, m3, m4)
    end
    @synchronize()
    lmax = max(ltmp[1], ltmp[65])
    acc_ll1 = exp(acc_ll1 - lmax)
    acc_ll2 = exp(acc_ll2 - lmax)
    ltmp[a] = acc_ll1 + acc_ll2
    dtmp[a] = acc_ll1 * acc_dd1 + acc_ll2 * acc_dd2
    @synchronize()
    @nexprs 6 j->begin
        if a % 2^j == 1 # a % {2, 4, 8, 16, 32, 64}
            ltmp[a] = ltmp[a] + ltmp[a + 2^(j-1)] # [a] + [a + {1,2,4,8,16,32}]
            dtmp[a] = dtmp[a] + dtmp[a + 2^(j-1)]
        end
        @synchronize()
    end
    if a == 1
        dist_out[p] = sqrt((dtmp[1] + dtmp[65]) / (ltmp[1] + ltmp[65]))
    end
end

function cell_distance!(dist_out, del_gpu, eta_gpu, vg_gpu; batchsize = 50000)
    comb, s = combinations(1:size(dist_out)[2], 2), [1, 1]
    lastsize = batchsize = min(batchsize, length(comb))
    pairs = Vector{Int}(undef, batchsize)
    idx = Vector{CartesianIndex{2}}(undef, batchsize)
    backend = get_backend(vg_gpu)
    alpha_gpu = allocate(backend, Float32, 256)
    copyto!(alpha_gpu, collect(1:256) ./ 128)
    pairs_gpu = allocate(backend, Int, batchsize)
    k1! = celldistance_256bins!(backend, (1, 128))
    dist_gpu = allocate(backend, Float32, batchsize)
    dist_tmp_cpu = Vector{Float32}(undef, batchsize)
    @time while !isnothing(s)
        for i in 1:batchsize
            s = iterate(comb, s)
            if isnothing(s)
                lastsize = i - 1
                break
            end
            x, s = s
            pairs[i] = x[1] << 32 | x[2]
            idx[i] = CartesianIndex(x[2], x[1]) # lower triangular
        end
        if lastsize == 0 break end
        ## begin GPU region
        copyto!(pairs_gpu, pairs)
        k1!(pairs_gpu, del_gpu, eta_gpu, vg_gpu, alpha_gpu, dist_gpu; ndrange = (lastsize, 128))
        copyto!(dist_tmp_cpu, dist_gpu)
        ## end GPU region
        dist_out[idx[1:lastsize]] .= dist_tmp_cpu[1:lastsize]
    end
    idx = findall(.!(isfinite.(dist_out))) # reevaluate any nan or inf result
    np = length(idx)
    if np > 0
        pairs = [x[1] << 32 | x[1] for x in idx]
        resize!(dist_gpu, np)
        resize!(dist_tmp_cpu, np)
        copyto!(pairs_gpu, pairs)
        k1!(pairs_gpu, del_gpu, eta_gpu, vg_gpu, alpha_gpu, dist_gpu; ndrange = (np, 128))
        copyto!(dist_tmp_cpu, dist_gpu)
        dist_out[idx] .= dist_tmp_cpu
    end
end
