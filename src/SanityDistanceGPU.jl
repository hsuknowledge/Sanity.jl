using Combinatorics
using KernelAbstractions

@inline mymax(x, y) = x > y ? x : y

@kernel cpu=false inbounds=true function celldistance_256bins!(
        @Const(pairs), @Const(delta), @Const(epsilon), @Const(vg), @Const(alpha), dist_out
    )
    p, a = @index(Global, NTuple) # pair & alpha indices
    ltmp = @localmem Float32 256
    dtmp = @localmem Float32 256
    i1, i2 = pairs[p] >>> 32, pairs[p] & 0xffffffff
    @private acc_ll = 0f0
    @private acc_dd = 0f0
    for g in 1:size(vg)[1]
        x2 = abs2(delta[g, i1] - delta[g, i2])
        e2 = epsilon[g, i1] + epsilon[g, i2]
        prior = inv(alpha[a] * vg[g] + e2)
        acc_ll += x2 * prior - log(prior)
        shrinker = alpha[a] * vg[g] * prior
        acc_dd += (x2 * shrinker + e2) * shrinker
    end
    acc_ll = reinterpret(Float32, reinterpret(UInt32, acc_ll) + 0x7f800000) # / -2
    ltmp[a] = acc_ll
    @synchronize()
    if a <= 128 ltmp[a] = mymax(ltmp[a], ltmp[a + 128]) end
    @synchronize()
    if a <= 64  ltmp[a] = mymax(ltmp[a], ltmp[a + 64])  end
    @synchronize()
    if a <= 32  ltmp[a] = mymax(ltmp[a], ltmp[a + 32])  end
    @synchronize()
    if a <= 16  ltmp[a] = mymax(ltmp[a], ltmp[a + 16])  end
    @synchronize()
    if a <= 8   ltmp[a] = mymax(ltmp[a], ltmp[a + 8])   end
    @synchronize()
    if a <= 4   ltmp[a] = mymax(ltmp[a], ltmp[a + 4])   end
    @synchronize()
    if a <= 2   ltmp[a] = mymax(ltmp[a], ltmp[a + 2])   end
    @synchronize()
    if a <= 1   ltmp[a] = mymax(ltmp[a], ltmp[a + 1])   end
    @synchronize()
    ltmp[a] = acc_ll = exp(acc_ll - ltmp[1])
    dtmp[a] = acc_ll * acc_dd
    @synchronize()
    if a <= 128
        ltmp[a] = ltmp[a] + ltmp[a + 128]
        dtmp[a] = dtmp[a] + dtmp[a + 128]
    end
    @synchronize()
    if a <= 64
        ltmp[a] = ltmp[a] + ltmp[a + 64]
        dtmp[a] = dtmp[a] + dtmp[a + 64]
    end
    @synchronize()
    if a <= 32
        ltmp[a] = ltmp[a] + ltmp[a + 32]
        dtmp[a] = dtmp[a] + dtmp[a + 32]
    end
    @synchronize()
    if a <= 16
        ltmp[a] = ltmp[a] + ltmp[a + 16]
        dtmp[a] = dtmp[a] + dtmp[a + 16]
    end
    @synchronize()
    if a <= 8
        ltmp[a] = ltmp[a] + ltmp[a + 8]
        dtmp[a] = dtmp[a] + dtmp[a + 8]
    end
    @synchronize()
    if a <= 4
        ltmp[a] = ltmp[a] + ltmp[a + 4]
        dtmp[a] = dtmp[a] + dtmp[a + 4]
    end
    @synchronize()
    if a <= 2
        ltmp[a] = ltmp[a] + ltmp[a + 2]
        dtmp[a] = dtmp[a] + dtmp[a + 2]
    end
    @synchronize()
    if a <= 1
        ltmp[a] = ltmp[a] + ltmp[a + 1]
        dtmp[a] = dtmp[a] + dtmp[a + 1]
    end
    @synchronize()
    dist_out[p] = sqrt(dtmp[1] / ltmp[1])
end

function cell_distance!(dist_out, del_gpu, eps_gpu, vg_gpu; batchsize = 10000)
    comb, s = combinations(1:size(dist_out)[2], 2), [1, 1]
    lastsize = batchsize
    nb = 256
    pairs = Vector{Int}(undef, batchsize)
    idx = Vector{CartesianIndex{2}}(undef, batchsize)
    backend = get_backend(vg_gpu)
    alpha_gpu = allocate(backend, Float32, nb)
    copyto!(alpha_gpu, collect(1:nb) ./ (nb / 2))
    pairs_gpu = allocate(backend, Int, batchsize)
    k1! = celldistance_256bins!(backend, (1, nb))
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
        k1!(pairs_gpu, del_gpu, eps_gpu, vg_gpu, alpha_gpu, dist_gpu; ndrange = (lastsize, nb))
        copyto!(dist_tmp_cpu, dist_gpu)
        ## end GPU region
        dist_out[idx[1:lastsize]] .= dist_tmp_cpu[1:lastsize]
    end
end
