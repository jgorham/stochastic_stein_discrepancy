# Multinomial Stein Distribution
#
# Represents a Multinomial distribution with parameters ntrials, probs

using Distributions: Multinomial
import Distributions

type SteinMultinomial <: SteinDistribution
    # Number of trials
    ntrials::Int64
    # Probability vector, sums to 1
    probs::Array{Float64}
    # Stein factors, length is 1 less than that of probs
    c1::Float64
    c2::Float64
    c3::Float64
end

# Constructor with (c1,c2,c3) = (1,1,1)
SteinMultinomial(ntrials::Int64, probs::Array{Float64}) = SteinMultinomial(
    ntrials,
    probs,
    1.0,
    1.0,
    1.0
)

# Constructor with (c1,c2,c3) = (1,1,1), all categories equiprobable
SteinMultinomial(ntrials::Int64, ncategories::Int64) = SteinMultinomial(
    ntrials,
    [1.0/ncategories for i=1:ncategories],
    1.0,
    1.0,
    1.0
)

# Draw n independent samples from distribution; last column is dropped
function rand(d::SteinMultinomial, n::Int64)
    Distributions.rand(Multinomial(d.ntrials,d.probs),n)'[:,1:end-1]
end

# Joint distribution evaluated at a point y
function jointdistribution(d::SteinMultinomial, y::Array{Float64})
    Distributions.pdf(Multinomial(d.ntrials,d.probs),[y,d.ntrials-sum(y)])
end

# Conditional distribution of jth coordinate of y, given all other coordinates,
# represented as a probability vector
function conddistribution(d::SteinMultinomial, y::Array{Float64,1}, j::Int64)
    conddist = [jointdistribution(d, [y[1:(j-1)], yj, y[(j+1):end]])
                for yj in 0:d.ntrials]
    conddist / sum(conddist)
end

function supportlowerbound(d::SteinMultinomial, j::Int64)
    0
end

function supportupperbound(d::SteinMultinomial, j::Int64)
    d.ntrials
end

function numdimensions(d::SteinMultinomial)
    length(d.probs) - 1
end

# Cumulative distribution function (only valid when X is univariate)
