# compare-hyperparameters-gmm-posterior
#
# This script compares the quality of samples generated from different
# step size hyperparameter settings of a Stochastic Gradient Langevin Dynamics
# sampler for a Gaussian mixture model posterior target distribution.

include("experiment_utils.jl")

# set the discrepancy
@parsestringcli discrepancytype "k" "discrepancytype" "inversemultiquadric"
# set the sampler
@parsestringcli sampler "q" "sampler" "sgld"
# the number of likelihood terms to sample (0 means all)
@parseintcli likelihood_n "l" "likelihoodn" 0
# thinning posterior samples to save computation
@parseintcli thinningn "t" "thinningn" 1
# how many likelihood terms
@parseintcli L "L" "likelihoodterms" 100
# how many samples
@parseintcli n "n" "numsamples" 1000
# should we use multiple cores?
@parseintcli numcores "n" "numcores" 1
# parallelize it
if (numcores > 1)
    if (numcores > CPU_CORES)
        error(@sprintf("Requested %d cores, only %d available.", numcores, CPU_CORES))
    end
    addprocs(numcores - 1)
    @assert (numcores == nprocs())
end

using Iterators: product
import SteinDistributions
using SteinDiscrepancy:
    SteinInverseMultiquadricKernel,
    ksd,
    gsd
using SteinSamplers:
    runsgld,
    runapproxslice

# Run one complete experiment per random number generator seed
seeds = 1:50
# Select a solver for graph Stein discrepancy optimization problem
#solver = "clp"
solver = "gurobi"
## Sampler settings for SGLD
distname = "gmm-posterior"
# Generate random dataset from GMM
truex = [0.0, 1.0] # Parameter vector used to generate dataset
data_generation_seed = 1
sigma2y = 2.0
y = @setseed data_generation_seed SteinDistributions.randgmm(truex, L; sigma2y=sigma2y)
# save the y values for later
save_json(
    Dict{Any, Any}("y" => y);
    dir="compare-hyperparameters-$(distname)-y",
    numsamples=L,
    x=string(truex)
)
# Create posterior distribution based on dataset
@eval @everywhere target = SteinDistributions.SteinGMMPosterior($y)
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
# set the kernel
kernel = nothing
if discrepancytype == "inversemultiquadric"
    kernel = SteinInverseMultiquadricKernel(0.5)
end
# the max number of likelihood evals to run for
# (if seed=7, epsilon=0.1, sampler=approxslice, L=100, n=1000, maxeval=147695)
# (if seed=7, epsilon=0.2, sampler=approxslice, L=100, n=2000, maxeval=152875)
maxevals = 147695
# the SGLD epsilons to run the experiemnt
sgld_epsilons = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
# the approx slice epsilons to run the experiemnt
approxslice_epsilons = [0.0, 1e-2, 1e-1]
# Fix mini-batch size
batchsize = 5

if sampler == "sgld"
    # the number of sweeps to make through the data
    numsweeps = round(Int, ceil(n / floor(SteinDistributions.numdatapoints(target)/batchsize)))
    epsilons = sgld_epsilons
    runsampler(x0,epsilon) = runsgld(target, x0;
                                     epsilonfunc=t -> epsilon,
                                     numsweeps=numsweeps,
                                     batchsize=batchsize)
elseif sampler == "approxslice"
    epsilons = approxslice_epsilons
    runsampler(x0,epsilon) = runapproxslice(target, x0;
                                            epsilon=epsilon,
                                            numlikelihood=maxevals,
                                            batchsize=batchsize)
end

## Draw and evaluate samples for different hyperparameter and seed combinations
d = SteinDistributions.numdimensions(target)
for (epsilon, seed) in product(epsilons, seeds)
    # Generate starting sample value x0
    x0 = @setseed seed randn(d)

    println("Generating sample for seed=$(seed), epsilon=$(epsilon)")
    # Run sampler
    points = []; numgrad = NaN
    @trycatchcontinue(
        begin
            (points, numgrad) = @setseed seed runsampler(x0,epsilon)
            if any(isnan(points))
                println("[$(distname):seed=$(seed),epsilon=$(epsilon)] NaNs found. Skipping.")
                continue
            end
        end,
        println("[$(distname):seed=$(seed),epsilon=$(epsilon)]:")
    )

    # thin the sample if necessary
    thinpoints = points[thinningn:thinningn:size(points,1),:]

    # Compute Stein discrepancy for first n points in sample
    # (with equal point weights)
    println("Computing Stein discrepancy for n=$(n), d=$(d), dist=$(distname)")
    objectivevalue = nothing
    edgetime = nothing
    solvetime = nothing

    if discrepancytype == "graph"
        res = nothing
        @trycatchcontinue(
            begin
                res = gsd(points=thinpoints, gradlogdensity=gradlogp, solver=solver)
            end,
            println("[n=$(n)|sampler=$(distname)]:")
        )

        objectivevalue = res.objectivevalue
        edgetime = res.edgetime
        solvetime = res.solvetime
    elseif kernel != nothing
        res = ksd(points=thinpoints, gradlogdensity=gradlogp, kernel=kernel)
        objectivevalue = sqrt(res.discrepancy2)
        solvetime = res.solvetime
    end

    println("\tn = $(n), objective = $(objectivevalue)")
    # Package results
    instance_data = Dict{Any, Any}(
        "distname" => distname,
        "n" => size(points, 1),
        "sampler" => sampler,
        "solver" => solver,
        "d" => d,
        "seed" => seed,
        "epsilon" => epsilon,
        "X" => points,
        "q" => res.weights,
        "objectivevalue" => objectivevalue,
        "likelihoodn" => likelihood_n,
        "thinningn" => thinningn,
        "numgrad" => numgrad,
        "edgetime" => edgetime,
        "solvetime" => solvetime,
        "ncores" => nprocs(),
    )

    # Save results
    save_json(
        instance_data;
        dir="compare-hyperparameters-$(distname)",
        distname=distname,
        discrepancytype=discrepancytype,
        sampler=sampler,
        n=n,
        solver=solver,
        epsilon=epsilon,
        ncores=nprocs(),
        likelihoodn=likelihood_n,
        thinningn=thinningn,
        L=L,
        d=d,
        seed=seed,
    )
end
