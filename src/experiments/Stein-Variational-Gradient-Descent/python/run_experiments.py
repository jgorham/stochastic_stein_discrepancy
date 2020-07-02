import argparse
import numpy as np
import time

from bayesian_nn import svgd_bayesnn

NUM_TRIALS = 20

def load_data(dataset):
    if dataset == 'boston':
        return np.loadtxt('../data/boston_housing')
    elif dataset == 'yacht':
        return np.loadtxt('../data/yacht_hydrodynamics.data')
    elif dataset == 'concrete':
        return np.loadtxt('../data/Concrete')
    elif dataset == 'real_estate':
        return np.loadtxt('../data/real_estate.tsv')
    elif dataset == 'naval':
        return np.loadtxt('../data/naval.txt')[:,:-1]
    elif dataset == 'energy':
        return np.loadtxt('../data/energy.tsv')[:,:-1]

if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description='Run SVGD for various datasets.')
    parser.add_argument('--dataset', help='the dataset to run SVGD on')
    parser.add_argument('--batch_size_frac', type=float, default=1.0, help='the batchsize fraction for SVGD')
    parser.add_argument('--n_hidden', type=int, default=50, help='the size of hidden layer')
    parser.add_argument('--particles', type=int, default=20, help='number of SVGD particles')
    parser.add_argument('--stepsize', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('--use-saga', action='store_true', help='whether to use SAGA for variance reduction')
    parser.add_argument('--get-checkpoints', action='store_true', help='whether to produce checkpoints or not')
    args = parser.parse_args()
    dataset, batch_size_frac, n_hidden, particles, use_saga, get_checkpoints, stepsize = (
        args.dataset, args.batch_size_frac, args.n_hidden, args.particles, args.use_saga, args.get_checkpoints, args.stepsize
    )
    # load data
    data = load_data(dataset)
    # Please make sure that the last column is the label and the other columns are features
    X_input = data[:, range(data.shape[1] - 1)]
    y_input = data[:, data.shape[1] - 1]
    train_ratio = 0.9  # We create the train and test sets with 90% and 10% of the data

    for trial in range(NUM_TRIALS):
        np.random.seed(trial)
        permutation = np.random.permutation(X_input.shape[0])

        size_train = int(np.round(X_input.shape[0] * train_ratio))
        index_train = permutation[0:size_train]
        index_test = permutation[size_train:]

        X_train, y_train = X_input[index_train, :], y_input[index_train]
        X_test, y_test = X_input[index_test, :], y_input[index_test]

        start = time.time()
        ''' Training Bayesian neural network with SVGD '''
        if get_checkpoints:
            svgd = svgd_bayesnn(X_train, y_train, batch_size_frac=batch_size_frac, n_hidden=n_hidden,
                                max_iter=2000, M=particles, use_saga=use_saga, master_stepsize=stepsize,
                                X_test=X_test, y_test=y_test)
            checkpoints = svgd.get_checkpoints()
            for checkpoint in checkpoints:
                print '\t'.join(str(x) for x in [dataset, trial, batch_size_frac, n_hidden, particles, int(use_saga), stepsize] + list(checkpoint))
        else:
            svgd = svgd_bayesnn(X_train, y_train, batch_size_frac=batch_size_frac, n_hidden=n_hidden,
                                max_iter=2000, M=particles, use_saga=use_saga, master_stepsize=stepsize)
            svgd_time = time.time() - start
            svgd_rmse, svgd_ll = svgd.evaluation(X_test, y_test)
            print '\t'.join(str(x) for x in [dataset, trial, batch_size_frac, n_hidden, particles, int(use_saga), stepsize, svgd_rmse, svgd_ll, svgd_time])
