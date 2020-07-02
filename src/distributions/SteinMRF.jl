# Markov random field of binary variables. Following
#
# http://arxiv.org/pdf/1304.5299v4.pdf,
#
# the joint distribution is
#
#   P(X) \propto \prod_{i < j < k} psi_{ijk}(x_i, x_j, x_k).
#
# The log potentials log psi_{ijk} are drawn iid from a normal distribution with
# mean 0 and variance 0.02. For convenience, we store log psi_{ijk} for all
# triples (i,j,k), not just those for which i < j < k. We then impose the
# symmetry constraints
#
#   psi_{ijk}(x_i, x_j, x_k) = psi_{ikj}(x_i, x_k, x_j)
#                            = psi_{kij}(x_k, x_i, x_j),
#
# and similarly for the other three possible permutations of i,j,k. Thus the
# joint distribution can also be written
#
#   P(X) \propto (\prod_{i ≠ j ≠ k} psi_{ijk}(x_i, x_j, x_k))^{1/6}.

type SteinMRF <: SteinDistribution
    # Dimensionality
    p::Int64
    # Array of log potentials, to be symmetrized in inner constructor
    logpsi::Array{Float64}
    # Stein factors
    c1::Float64
    c2::Float64
    c3::Float64
    SteinMRF(p::Int64, logpsi::Array{Float64}, c1::Float64,
             c2::Float64, c3::Float64) = (
        logpsi_ = symmetrize(p, logpsi);
        return new(p, logpsi_, c1, c2, c3);
    )
end

function symmetrize(p::Int64, logpsi::Array{Float64})
    for i = 1:p, j = 1:p, k = 1:p
        idx = sortperm([i,j,k])
        for l = 1:2, m = 1:2, n = 1:2
            logpsi[i,j,k,l,m,n] = logpsi[[i,j,k][idx]...,[l,m,n][idx]...];
        end
    end
    return logpsi
end

# Constructor with (c1,c2,c3) = (1,1,1)
SteinMRF(p::Int64) = SteinMRF(
    p,
    sqrt(0.02)*randn(p,p,p,2,2,2),
    1.0,
    1.0,
    1.0
);

# Evaluate log odds
#   log P(y_i = 1 | y_{-i}) - log P(y_i = 0 | y_{-i})
# by summing over (j,k) such that j < k, j ≠ i, and k ≠ i.
function condlogodds(d::SteinMRF, y::Array{Float64,1}, i::Int64)
    logodds = 0.0
    for j = setdiff(1:d.p, i)
        for k = setdiff((j+1):d.p, i)
            logodds += d.logpsi[i,j,k,1+1,y[j]+1,y[k]+1]
            logodds -= d.logpsi[i,j,k,0+1,y[j]+1,y[k]+1]
        end
    end
    logodds
end

# Conditional distribution of jth coordinate of y, given all other coordinates,
# represented as a probability vector
function conddistribution(d::SteinMRF, y::Array{Float64,1}, j::Int64)
    p1 = exp(condlogodds(d, y, j))
    p1 = p1 / (1 + p1)
    [1-p1, p1]
end

function supportlowerbound(d::SteinMRF, j::Int64)
    0
end

function supportupperbound(d::SteinMRF, j::Int64)
    1
end

function numdimensions(d::SteinMRF)
    d.p
end
