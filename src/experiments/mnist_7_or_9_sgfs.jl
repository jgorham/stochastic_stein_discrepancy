# mnist_7_or_9_sgfs
#
# This script compares the Stein discrepancy of sample sequences drawn
# from SGFS as done in Experiment 5.1 of (http://arxiv.org/abs/1206.6380).
# We got the Matlab code from the authors and have used their
# code to re-generate the same samples as used in the paper.
# By plotting the Stein discrepancy results against
# a measure of computation required per sample (e.g., the number of
# datapoint likelihood evaluations needed), we can quantify
# which of the samplers is indeed best for representing the target.
#
# The target distribution is a Bayesian logistic regression posterior
# under independent Gaussian priors based on a the mnist dataset
# with either 7 or 9 as the labels.

# parallelize now!
addprocs(Sys.CPU_CORES - 1)

using DataFrames

import SteinDistributions
using SteinDiscrepancy:
    ksd,
    SteinGaussianKernel,
    SteinGaussianPowerKernel,
    SteinMaternRadialKernel,
    SteinInverseMultiquadricKernel

BASEPATH = "src/experiments/mnist"

include("experiment_utils.jl")

# Set random number generator seed
@parseintcli seed "s" "seed" 7
# set the sampler
@parsestringcli sampler "r" "sampler" "SGFS-f"
# the number of likelihood terms to sample (0 means all)
@parseintcli likelihood_n "l" "likelihoodn" 0
# thinning posterior samples to save computation
@parseintcli thinningn "t" "thinningn" 1
# set the kernel
@parsestringcli kernelname "k" "kernel" "inversemultiquadric"
kernel = nothing
# Setup the data
data_set = "mnist"
Xdat = readtable(joinpath(BASEPATH, "mnist79Xtrain50proj.csv"), header=false)
Xtrain = Array(Float64, nrow(Xdat), ncol(Xdat))
for jj in 1:ncol(Xdat)
    Xtrain[:,jj] = collect(skipmissing(Xdat[:,jj]))
end
ydat = readtable(joinpath(BASEPATH, "mnist79ytrain50proj.csv"), header=false)
ytrain = collect(skipmissing(ydat[:,1]))
# get the training sizes
ntrain = size(Xtrain, 1)
# give X an intercept term
X = hcat(Xtrain, ones(ntrain))
# make y be -1 or 1
y = vec(2.0 * ytrain - 1.0)
# Parameter dimension
d = size(X, 2)
# name of prior + posterior
distname = "logisticgaussianprior"
# sigma2 term for prior
sigma2 = 1.0
# setup distribution
@eval @everywhere target = SteinDistributions.SteinLogisticRegressionGaussianPrior($X, $y, zeros($d), $sigma2 .* ones($d))
@eval @everywhere likelihoodn = $likelihood_n
# define gradlogp
if likelihood_n == 0
    @everywhere function gradlogp(x::Array{Float64,1})
        SteinDistributions.gradlogdensity(target, x)
    end
else
    @everywhere function gradlogp(x::Array{Float64,1})
        L = SteinDistributions.numdatapoints(target)
        SteinDistributions.gradlogdensity(target, x; idx=randperm(L)[1:likelihoodn])
    end
end
# number of burnin samples
b = 50000

if sampler in ["SGFS-d", "SGFS-f"]
    sample_filename = joinpath(
        BASEPATH,
        @sprintf("mnist79_%s_n=100000.csv", sampler)
    )
    betadat = readtable(sample_filename, header=false)
    betas = Array(Float64, nrow(betadat), ncol(betadat))
    for jj in 1:ncol(betadat)
        betas[:,jj] = collect(skipmissing(betadat[:,jj]))
    end
    betas = betas[(b+1):end,:]
    n = size(betas,1)
    # thin the sample if necessary
    thinbetas = betas[thinningn:thinningn:size(betas,1),:]
    # checkpoints for the kernel
    checkpoints = vcat(100:100:min(1000,n),
                       2000:1000:min(n,10000),
                       20000:10000:min(n,100000))
    if checkpoints[end] != n
        push!(checkpoints, n)
    end
    # get the kernel
    if kernelname == "gaussian"
        kernel = SteinGaussianKernel()
    elseif kernelname == "gaussianpower"
        kernel = SteinGaussianPowerKernel(2.0)
    elseif kernelname == "maternradial"
        kernel = SteinMaternRadialKernel()
    elseif kernelname == "inversemultiquadric"
        kernel = SteinInverseMultiquadricKernel(0.5)
    end
    # Compute Stein discrepancy
    res = @setseed seed ksd(
        points=thinbetas,
        gradlogdensity=gradlogp,
        kernel=kernel,
        checkpoints=checkpoints)

    betahat = vec(mean(betas, 1))
    sigmahat = cov(betas)

    instance_data = Dict{Any, Any}(
        "distname" => distname,
        "n" => checkpoints,
        "d" => d,
        "betahat" => betahat,
        "sigmahat" => sigmahat,
        "kernel" => kernelname,
        "likelihoodn" => likelihood_n,
        "thinningn" => thinningn,
        "betas" => betas,
        "discrepancy2" => res.discrepancy2,
        "solvetime" => res.solvetime,
        "sampler" => sampler,
        "ncores" => nprocs(),
        "burnin" => b,
    )

    save_json(
        instance_data;
        dir="mnist_7_or_9_sgfs",
        distname=distname,
        dataset=data_set,
        kernel=kernelname,
        sampler=sampler,
        likelihoodn=likelihood_n,
        thinningn=thinningn,
        n=n,
        d=d,
        ncores=nprocs()
    )
end
