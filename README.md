# Stochastic Stein Discrepancies

This repo contains many experiments that utilize the *Stochastic Stein
discrepancy*.

# Requirements

The first two experiments are written in Julia v0.6.
There is a file `src/startup.jl` which simply adds the paths of some modules
to the `LOAD_PATH` variable.  You should either simlink this file to
`~/.juliarc.jl` or add the lines from this file to your current
`.juliarc.jl`.  There is a REQUIRE file that demarcates all the necessary
Julia packages needed to run all the experiments; these can be added by
runnning inside a Julia repl

```
Pkg.add("<package_name>")
```

The third experiment (SSVGD) is a fork of an existing Python repo. In order
to install the requirements for that experiment, please first install
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) and then run the
following commands:

```
cd src/experiments/Stein-Variational-Gradient-Descent/
conda create --name svgd python=2.7
source activate svgd
pip install -r requirements.txt
```

This will prepare your python environment. Run

```
source decativate
```

to exit the virtual Python env.

# Training

Below outlines how to run each experiment:

## Experiment 1: (Hyperparameter selection for approximate MCMC)

This command should be run from the base directory of this repo.  To
generate the data for the first experiment, one should run

```
julia src/experiments/compare-hyperparameters-gmm-posterior.jl --likelihoodn=<n>
```

where n is 0, 1, and 10. This will dump artifacts in the results directory.

## Experiment 2: Selecting biased MCMC samplers

This command should be run from the base directory of this repo.  To
generate the data for the first experiment, one should run

```
julia src/experiments/mnist_7_or_9_sgfs.jl --sampler=<sampler> --likelihoodn=<n>
```

where (sample, n) belongs to {SGFS-f, SGFS-d} x {0, 1000, 100}. This will
dump artifacts in the results directory.

## Experiment 3: Improving particle approximations with SSVGD

This is the only experiment to be run using Python2. From the base directory, run

```
mkdir -p results/stochastic_svgd/data/
cd src/experiments/Stein-Variational-Gradient-Descent/python
```

Assuming you have already activated the conda environment with

```
source activate svgd
```

then one can kick off the experiments by running

```
python run_experiments.py --particles=20 --n_hidden=50 --dataset=<dataset> --batch_size_frac=<b> --stepsize=<eps> --get-checkpoints > ../../../../results/stochastic_svgd/data/svgd_dataset=<dataset>_batchsizefrac=<b>_nhidden=50_particles=20_stepsize=<eps>.tsv
```

where b is chosen from {0.1, 0.25, 1.0} and (dataset, eps) are chosen from
{(yacht, 1e-2), (boston, 1e-3), (naval, 1e-3)}. This will generate artifacts
in the proper location to make visualization easier.

# Evaluation

All scripts to plot the results can be found in the `src/visualization` directory.
The version of R used was v3.4.2. These can be run via Rscript from inside the
`src/visualization` directory, e.g.,

```
cd src/visualization
Rscript stochastic-checkpoint-svgd-comparison_viz.R
```