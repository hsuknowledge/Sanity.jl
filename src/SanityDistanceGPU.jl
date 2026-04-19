using Combinatorics
using KernelAbstractions

@inline mymax(x, y) = x > y ? x : y

@kernel cpu=false inbounds=true function celldistance_256bins!(
        @Const(pairs), @Const(del), @Const(eta), @Const(vg), @Const(alpha), dist_out
    )
    log2e = reinterpret(Float32, 0x3fb8aa3b)
    p, a = @index(Global, NTuple) # pair & alpha indices
    ltmp = @localmem Float32 128
    dtmp = @localmem Float32 128
    i1, i2 = pairs[p] >>> 32, pairs[p] & 0xffffffff
    ng = size(vg)[1]
    @private acc_ll1 = 0f0
    @private acc_ll2 = 0f0
    @private acc_dd1 = 0f0
    @private acc_dd2 = 0f0
    for g in 1:fld(ng, 4)
        g1, g2, g3, g4 = 4g-3, 4g-2, 4g-1, 4g
        x2_1 = abs2(del[g1, i1] - del[g1, i2])
        x2_2 = abs2(del[g2, i1] - del[g2, i2])
        x2_3 = abs2(del[g3, i1] - del[g3, i2])
        x2_4 = abs2(del[g4, i1] - del[g4, i2])
        e2_1 = eta[g1, i1] + eta[g1, i2]
        e2_2 = eta[g2, i1] + eta[g2, i2]
        e2_3 = eta[g3, i1] + eta[g3, i2]
        e2_4 = eta[g4, i1] + eta[g4, i2]
        @fastmath p1_1 = inv(alpha[2a-1] * vg[g1] + e2_1)
        @fastmath p1_2 = inv(alpha[2a-1] * vg[g2] + e2_2)
        @fastmath p1_3 = inv(alpha[2a-1] * vg[g3] + e2_3)
        @fastmath p1_4 = inv(alpha[2a-1] * vg[g4] + e2_4)
        @fastmath p2_1 = inv(alpha[2a] * vg[g1] + e2_1)
        @fastmath p2_2 = inv(alpha[2a] * vg[g2] + e2_2)
        @fastmath p2_3 = inv(alpha[2a] * vg[g3] + e2_3)
        @fastmath p2_4 = inv(alpha[2a] * vg[g4] + e2_4)
        @fastmath acc_ll1 += x2_1 * p1_1 * log2e - log2(p1_1)
        @fastmath acc_ll1 += x2_2 * p1_2 * log2e - log2(p1_2)
        @fastmath acc_ll1 += x2_3 * p1_3 * log2e - log2(p1_3)
        @fastmath acc_ll1 += x2_4 * p1_4 * log2e - log2(p1_4)
        @fastmath acc_ll2 += x2_1 * p2_1 * log2e - log2(p2_1)
        @fastmath acc_ll2 += x2_2 * p2_2 * log2e - log2(p2_2)
        @fastmath acc_ll2 += x2_3 * p2_3 * log2e - log2(p2_3)
        @fastmath acc_ll2 += x2_4 * p2_4 * log2e - log2(p2_4)
        s1_1 = alpha[2a-1] * vg[g1] * p1_1
        s1_2 = alpha[2a-1] * vg[g2] * p1_2
        s1_3 = alpha[2a-1] * vg[g3] * p1_3
        s1_4 = alpha[2a-1] * vg[g4] * p1_4
        s2_1 = alpha[2a] * vg[g1] * p2_1
        s2_2 = alpha[2a] * vg[g2] * p2_2
        s2_3 = alpha[2a] * vg[g3] * p2_3
        s2_4 = alpha[2a] * vg[g4] * p2_4
        @fastmath acc_dd1 += (x2_1 * s1_1 + e2_1) * s1_1
        @fastmath acc_dd1 += (x2_2 * s1_2 + e2_2) * s1_2
        @fastmath acc_dd1 += (x2_3 * s1_3 + e2_3) * s1_3
        @fastmath acc_dd1 += (x2_4 * s1_4 + e2_4) * s1_4
        @fastmath acc_dd2 += (x2_1 * s2_1 + e2_1) * s2_1
        @fastmath acc_dd2 += (x2_2 * s2_2 + e2_2) * s2_2
        @fastmath acc_dd2 += (x2_3 * s2_3 + e2_3) * s2_3
        @fastmath acc_dd2 += (x2_4 * s2_4 + e2_4) * s2_4
    end
    for g in (fld(ng, 4)*4+1):ng
        x2 = abs2(del[g, i1] - del[g, i2])
        e2 = eta[g, i1] + eta[g, i2]
        @fastmath p1 = inv(alpha[2a-1] * vg[g] + e2)
        @fastmath p2 = inv(alpha[2a] * vg[g] + e2)
        @fastmath acc_ll1 += x2 * p1 * log2e - log2(p1)
        @fastmath acc_ll2 += x2 * p2 * log2e - log2(p2)
        s1 = alpha[2a-1] * vg[g] * p1
        s2 = alpha[2a] * vg[g] * p2
        @fastmath acc_dd1 += (x2 * s1 + e2) * s1
        @fastmath acc_dd2 += (x2 * s2 + e2) * s2
    end
    acc_ll1 *= -0.5f0
    acc_ll2 *= -0.5f0
    ltmp[a] = mymax(acc_ll1, acc_ll2)
    @synchronize()
    if a <= 64
        ltmp[a] = mymax(ltmp[a], ltmp[a + 64])
    end
    @synchronize()
    if a <= 32
        ltmp[a] = mymax(ltmp[a], ltmp[a + 32])
    end
    @synchronize()
    if a <= 16
        ltmp[a] = mymax(ltmp[a], ltmp[a + 16])
    end
    @synchronize()
    if a <= 8
        ltmp[a] = mymax(ltmp[a], ltmp[a + 8])
    end
    @synchronize()
    if a <= 4
        ltmp[a] = mymax(ltmp[a], ltmp[a + 4])
    end
    @synchronize()
    if a <= 2
        ltmp[a] = mymax(ltmp[a], ltmp[a + 2])
    end
    @synchronize()
    lmax = mymax(ltmp[1], ltmp[2])
    acc_ll1 = 2^(acc_ll1 - lmax)
    acc_ll2 = 2^(acc_ll2 - lmax)
    ltmp[a] = acc_ll1 + acc_ll2
    dtmp[a] = acc_ll1 * acc_dd1 + acc_ll2 * acc_dd2
    @synchronize()
    if a % 2 == 1
        ltmp[a] = ltmp[a] + ltmp[a + 1]
        dtmp[a] = dtmp[a] + dtmp[a + 1]
    end
    @synchronize()
    if a % 4 == 1
        ltmp[a] = ltmp[a] + ltmp[a + 2]
        dtmp[a] = dtmp[a] + dtmp[a + 2]
    end
    @synchronize()
    if a % 8 == 1
        ltmp[a] = ltmp[a] + ltmp[a + 4]
        dtmp[a] = dtmp[a] + dtmp[a + 4]
    end
    @synchronize()
    if a % 16 == 1
        ltmp[a] = ltmp[a] + ltmp[a + 8]
        dtmp[a] = dtmp[a] + dtmp[a + 8]
    end
    @synchronize()
    if a % 32 == 1
        ltmp[a] = ltmp[a] + ltmp[a + 16]
        dtmp[a] = dtmp[a] + dtmp[a + 16]
    end
    @synchronize()
    if a % 64 == 1
        ltmp[a] = ltmp[a] + ltmp[a + 32]
        dtmp[a] = dtmp[a] + dtmp[a + 32]
    end
    @synchronize()
    dist_out[p] = sqrt((dtmp[1] + dtmp[65]) / (ltmp[1] + ltmp[65]))
end

function cell_distance!(dist_out, del_gpu, eta_gpu, vg_gpu; batchsize = 10000)
    comb, s = combinations(1:size(dist_out)[2], 2), [1, 1]
    lastsize = batchsize
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
