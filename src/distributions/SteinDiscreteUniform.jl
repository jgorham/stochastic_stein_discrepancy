# Discrete Uniform Stein Distribution
#
# Represents a uniform distribution with p independent components taking on
# integer values between 0 and ulim

type SteinDiscreteUniform <: SteinDistribution
    # Marginals are DUnif(0,ulim)
    ulim::Int64
    # Dimensionality
    p::Int64
    # Stein factors
    c1::Float64
    c2::Float64
    c3::Float64
end

# Constructor with (c1,c2,c3) = (1,1,1)
SteinDiscreteUniform(ulim::Int64, p::Int64) = SteinDiscreteUniform(
    ulim,
    p,
    1.0,
    1.0,
    1.0
);

# Draw n independent samples from distribution
function rand(d::SteinDiscreteUniform, n::Int64)
    p = d.p;
    # Sample DUnif(0,ulim) variables
    rand(0:d.ulim, n, p)
end

# Joint distribution evaluated at a point y (unnormalized)
function jointdistribution(d::SteinDiscreteUniform, y::Array{Float64})
    1.0
end

# Conditional distribution of jth coordinate of y, given all other coordinates,
# represented as a probability vector
function conddistribution(d::SteinDiscreteUniform, y::Array{Float64,1}, j::Int64)
    ones(d.ulim+1) / (d.ulim+1)
end

function supportlowerbound(d::SteinDiscreteUniform, j::Int64)
    0
end

function supportupperbound(d::SteinDiscreteUniform, j::Int64)
    d.ulim
end

# Cumulative distribution function (only valid when X is univariate)
function cdf(d::SteinDiscreteUniform, t)
    if t >= d.ulim
        return 1.0
    elseif t < 0
        return 0.0
    else
        return (floor(t)+1.0)/(d.ulim+1.0)
    end
end

function numdimensions(d::SteinDiscreteUniform)
    d.p
end
