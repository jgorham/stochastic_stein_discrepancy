# Chain Potts Stein Distribution
#
# Represents a chain Potts distribution. Components are marginally +1 or -1
# with equal probability. The probability of (x_1,...x_p) is proportional to
#   exp(-beta * (x_1 x_2 + ... + x_{p-1} x_p)).
# The normalizing constant is 2*(2 cosh beta)^(p-1). beta is an affinity
# parameter; if beta > 0, then adjacent nodes tend to have opposite spins,
# and the opposite is true for beta < 0.
#
# We encode -1 as 0.

using StatsBase: wsample

type SteinPotts <: SteinDistribution
    # Affinity parameter
    beta::Float64
    # Dimensionality (number of nodes in the chain)
    p::Int64
    # Stein factors
    c1::Float64
    c2::Float64
    c3::Float64
end

# Constructor with (c1,c2,c3) = (1,1,1)
SteinPotts(beta::Float64, p::Int64) = SteinPotts(
    beta,
    p,
    1.0,
    1.0,
    1.0
);

# Draw n independent samples from distribution
function rand(d::SteinPotts, n::Int64)
    p = d.p;
    beta = d.beta;
    x = zeros(n,p);
    x[:,1] = rand(0:1,n);
    for i=1:n
        for j=2:p
            if x[i,j-1] == 1
                x[i,j] = wsample([0,1],[exp(beta),exp(-beta)])
            else
                x[i,j] = wsample([0,1],[exp(-beta),exp(beta)])
            end
        end
    end
    x
end

# Joint distribution evaluated at a single point
function jointdistribution(d::SteinPotts, y::Array{Float64})
    y = 2*y-1;     # Convert to +1 and -1
    exp(-d.beta*dot(y[1:d.p-1],y[2:d.p]))
end

# Conditional distribution of jth coordinate of y, given all other coordinates,
# represented as a probability vector
function conddistribution(d::SteinPotts, y::Array{Float64,1}, j::Int64)
    conddist = [jointdistribution(d, [y[1:(j-1)], yj, y[(j+1):end]])
                for yj in 0:1]
    conddist / sum(conddist)
end

function supportlowerbound(d::SteinPotts, j::Int64)
    0
end

function supportupperbound(d::SteinPotts, j::Int64)
    1
end

function numdimensions(d::SteinPotts)
    d.p
end
