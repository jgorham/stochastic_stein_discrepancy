# SteinProbitRegressionGaussianPrior
#
# We assume we have the following model:
#
# y_i = \Psi (beta_1 x_1 + ... + beta_p x_p)
#
# where \Psi is assumed to be the noraml CDF (and
# a Gaussian prior for beta).

using Distributions: Normal
import Distributions

type SteinProbitRegressionGaussianPrior <: SteinPosterior
    # the X data
    X::Array{Float64,2}
    # the 0/1 y values
    y::Array{Float64,1}
    # priorsigma2 is prior variance
    priorsigma2::Float64
end

defaultsigma2 = 0.1

SteinProbitRegressionGaussianPrior(X, y) =
    SteinProbitRegressionGaussianPrior(X, y, defaultsigma2)

# Returns lower bound of support
function supportlowerbound(d::SteinProbitRegressionGaussianPrior, i::Int)
    -Inf
end

# Returns upper bound of support
function supportupperbound(d::SteinProbitRegressionGaussianPrior, i::Int)
    Inf
end

# Returns the log prior log pi(beta)
function logprior(d::SteinProbitRegressionGaussianPrior,
                  beta::Array{Float64, 1})
    logpriorvalue =
        -0.5 * sum((beta .^ 2) ./ d.priorsigma2) -
        -0.5 * sum(log(2 * pi .* d.priorsigma2))
    logpriorvalue
end

# Returns log pi(d.y[idx] | beta)
function loglikelihood(d::SteinProbitRegressionGaussianPrior,
                       beta::Array{Float64, 1};
                       idx=1:length(d.y))
    y = d.y[idx]
    X = d.X[idx,:]
    ypos = find(y .> 0.0)
    yneg = find(y .<= 0.0)
    u = vec(X * beta)
    gsn = Normal()

    logprobpos = Distributions.logcdf(gsn, u[ypos])
    logprobneg = log(1.0 - Distributions.cdf(gsn, u[yneg]))
    sum(logprobpos) + sum(logprobneg)
end

# Returns the log prior log pi(beta)
function gradlogprior(d::SteinProbitRegressionGaussianPrior,
                      beta::Array{Float64, 1})
    -beta ./ d.priorsigma2
end

# Returns grad_{beta} log pi(d.y[idx] | beta)
function gradloglikelihood(d::SteinProbitRegressionGaussianPrior,
                           beta::Array{Float64,1};
                           idx=1:length(d.y))
    y = d.y[idx]
    X = d.X[idx,:]
    ypos = find(y .> 0.0)
    yneg = find(y .<= 0.0)
    u = vec(X * beta)
    gsn = Normal()

    posweights = Distributions.pdf(gsn, u[ypos]) ./ Distributions.cdf(gsn, u[ypos])
    negweights = -Distributions.pdf(gsn, u[yneg]) ./ (1.0 - Distributions.cdf(gsn, u[yneg]))

    xpos = vec(sum(broadcast(*, X[ypos,:], posweights), 1))
    xneg = vec(sum(broadcast(*, X[yneg,:], negweights), 1))

    xneg + xpos
end

function numdimensions(d::SteinProbitRegressionGaussianPrior)
    size(d.X, 2)
end
