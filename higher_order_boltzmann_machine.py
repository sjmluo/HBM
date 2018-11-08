# Author: Simon Luo

import numpy as np

import os

import time

try:
   import cPickle as pickle
except:
   import pickle

class Higher_Order_Boltzmann_Machine():
    """
    A memory and runtime efficient implementation of the Higher-Order Boltzmann Machine.
    This implementation uses an annuealed importance sampling (AIS) to approximate the
    partition function.

    Functions:
    ----------
    train
    set_emp_eta
    get_theta_sum
    theta2p
    get_model_predictions_by_gibbs_sample
    get_emperical_p
    ais2p
    save
    load

    Example:
    --------
    X = [[1,0,0,0],
         [0,0,0,1],
         [0,0,1,1],
         [1,1,1,1],
         [1,0,0,0],
         [0,0,0,1],
         [0,0,1,1],
         [1,1,1,1],
         [0,1,1,0],
         [0,1,1,1],
         [0,1,1,1],
         [0,0,1,0],
         [0,0,1,1],
         [0,1,0,0],
         [0,1,0,1],
         [1,0,1,1],
         [1,1,1,1]]
    X = np.array(X)
    HBM = Higher_Order_Boltzmann_Machine(X,order=2)
    HBM.train(lr=1, N_gibbs_samples=10000, burn_in=3, MAX_ITERATIONS=100, verbose=True)
    predict_C = [[0,1,0,1],
                 [1,0,1,1]]
    predict_C = np.array(predict_C)
    p_vec, p_lower_vec, p_upper_vec = HBM.ais2p(predict_C)
    """

    def __init__(self, X=None, order=1):
        """
        Initalise all variables required for the model.

        Parameters:
        -----------
        X: numpy.ndarray
            NxD matrix. Where N is the number of samples and D is the number
            of features. The features and labels are joined together in the
            same matrix. Where the first half of the matrix is the features
            and the second half is the labels.
        order: int
            Order of interaction for the Boltzmann Machine.

        Returns:
        --------
        None
        """
        self.order = order
        if X is None:
            return
        self.X = np.array(X, dtype=bool)

        self.log_N_max = self.X.shape[1]*np.log(2)

        self._get_nodes()

        self.theta_vec = np.zeros(self.C_vec.shape[0])
        self.theta_vec[0] = -self.log_N_max

        self.emp_eta_vec = self._compute_eta(self.X, self.C_vec)

        self.iterations = 0
        self.error = 0

    def _activate_nodes(self, C, ci, N, depth):
        """
        A recursive function to generate only the nodes which
        are required.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        if depth == 0:
            self.C_vec.append(C)
            return

        for c in range(ci, N-depth+1):
            C_current = np.array(C, dtype=bool)
            C_current[c] = True
            C_current = self._activate_nodes(C_current, c+1, N, depth-1)

    def _get_nodes(self):
        """
        Initalise nodes for the model.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        N = self.X.shape[1]
        self.C_vec = list()
        for d in range(self.order+1):
            self._activate_nodes(np.zeros(N, dtype=bool), 0, N, d)
        self.C_vec = np.array(self.C_vec, dtype=bool)

    def _compute_eta(self, X, C_vec):
        """
        Approximate eta from X.

        Parameters:
        -----------
        X: numpy.ndarray
            Sampled data points.
        C: numpy.ndarray
            Values of eta which we are interested in.
        Returns:
        --------
        eta_vec: numpy.ndarray
            A 1 dimensional vector of eta for C_vec.
        """
        C_vec = np.array(C_vec, dtype=bool)
        N = X.shape[0]
        eta_vec = np.array([np.sum(np.sum(X & C, axis=1) == np.sum(C)) / N
                            for C in C_vec])
        return eta_vec

    def _compute_p_from_x(self, X, C_vec):
        """
        Approximate p from X.

        Parameters:
        -----------
        X: numpy.ndarray
            Sampled data points.
        C: numpy.ndarray
            Values of eta which we are interested in.

        Returns:
        --------
        p_vec: numpy.ndarray
            A 1 dimensional vector of p for C_vec.
        """
        C_vec = np.array(C_vec, dtype=bool)
        N = X.shape[0]
        p_vec = np.array([np.sum(np.sum(X == C, axis=1) == len(C)) / N
                            for C in C_vec])
        return p_vec

    def _get_model_samples(self, N_gibbs_samples=10000, burn_in=3):
        """
        Apply Gibbs sampling on theta to find new X.

        Parameters:
        -----------
        N_gibbs_samples: int
            Number of data points used for Gibbs sampling
        burn_in: int
            Number of samples before using the data point

        Returns:
        --------
        X_samples: numpy.ndarray
        """
        D = self.X.shape[1]
        D_range = np.arange(D)
        samples = np.random.randint(0, self.X.shape[0], size=N_gibbs_samples)
        X_samples = np.array(self.X[samples,:], dtype=bool)
        # X_samples = np.random.rand(N_gibbs_samples, D) > 0.5

        # Begin Gibbs sampling
        for i in range(burn_in):
            # Iterate through each feature
            for d in np.random.permutation(D_range):
                # Get all values for fixed samples
                feature_idx = D_range != d
                sample_values = np.unique(X_samples[:, feature_idx], axis=0)

                # Iterate through fixed samples
                for si in sample_values:
                    # Index for samples to update
                    update_idx = np.sum(X_samples[:, feature_idx] == si, axis=1) == (D-1)

                    # Get combinations for active and inactive node
                    C = np.insert(si, d, True)
                    idx = (self.C_vec[:,d] == True) & (np.sum(self.C_vec & ~C, axis=1) == 0)
                    p = np.exp(np.sum(self.theta_vec[idx]))
                    true_p = p / (1 + p)

                    # Assign sampled value
                    draw_samples_idx = np.random.binomial(1, true_p, np.sum(update_idx)) == 1
                    X_samples[update_idx, d] = draw_samples_idx

        return X_samples

    def _gradient_descent_step(self, lr=0.1, N_gibbs_samples=10000, burn_in=3):
        """
        Single step for gradient descent

        Parameters:
        -----------
        lr: int
            Learning rate
        N_gibbs_samples: int
            Number of data points used for Gibbs sampling
        burn_in: int
            Number of samples before using the data point

        Returns:
        --------
        error: float
            Error from the gradient descent step
        """
        
        self.X_sampled = self._get_model_samples(N_gibbs_samples=N_gibbs_samples, burn_in=burn_in)
        eta_vec = self._compute_eta(self.X_sampled, self.C_vec)
        gradient = self.emp_eta_vec - eta_vec

        self.theta_vec += lr * gradient
        error = np.sum(gradient**2)**0.5
        return error


    def train(self, lr=1, N_gibbs_samples=10000, burn_in=3, MAX_ITERATIONS=1000, verbose=False, verbose_step=100, checkpoint_iteration=0, checkpoint_fname=None):
        """
        This function trains the model by matching the marginal probability distribution
        of the model with the emperical distribution.

        Parameters:
        -----------
        lr: int
            Learning rate
        N_gibbs_samples: int
            Number of data points used for Gibbs sampling
        burn_in: int
            Number of samples before using the data point
        MAX_ITERATIONS: int
            The maximum number of iterations
        verbose: bool
            Value should be True if status of the model is printed to screen.
        verbose_step: int
            The number of iterations before the status is printed.
        checkpoint_iteration: int
            Number of iterations before checkpoint is saved. No checkpoint is saved
            if the input value is 0.
        checkpoint_fname: str
            Filename of the checkpoint. If no value is input, then the time and date
            is used as the filename.

        Returns:
        --------
        None
        """
        checkpoint_path = './checkpoints/'
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if checkpoint_fname is None:
            time_string = time.strftime("%Y%m%d-%H%M%S")
            checkpoint_path += time_string
        else:
            checkpoint_path += checkpoint_fname


        beta = np.linspace(0,1,MAX_ITERATIONS+1)
        self._ais_init(beta, n_samples=N_gibbs_samples)
        while True:
            self.error = self._gradient_descent_step(lr=lr, N_gibbs_samples=N_gibbs_samples, burn_in=burn_in)
            self._ais_step(self.X_sampled)
            self.iterations += 1

            if verbose:
                if (self.iterations % verbose_step) == 0:
                    print('HBM', 'iterations:', self.iterations, 'error:', self.error)

            if checkpoint_iteration != 0:
                if (self.iterations % checkpoint_iteration) == 0:
                    self._update_nodes()
                    current_checkpoint_path = checkpoint_path + '_{}.pickle'.format(self.iterations)
                    self.save(current_checkpoint_path)

            if self.iterations >= MAX_ITERATIONS:
               print('error:', self.error)
               break
        log_Z_n, _, _ = self._ais_evaluate()
        self.theta_vec[0] = log_Z_n

    def set_emp_eta(self, C_vec, p_vec):
        """
        This function sets the target value for eta.
        This function is useful to calculate the model's mle.

        Parameters:
        -----------
        C_vec: numpy.ndarray
            An array of active nodes. Where the column represents
            the nodes. The rows represents the data points.
        p_vec: The probability of each element of C_vec.

        Returns:
        --------
        None
        """
        C_vec = np.array(C_vec, dtype=bool)
        for i, C in enumerate(self.C_vec):
            idx = np.sum(C_vec & C, axis=1) == np.sum(C)
            self.emp_eta_vec[i] = np.sum(p_vec[idx])

    def get_theta_sum(self, C_vec):
        """
        Returns sum of theta for C.

        Parameters:
        -----------
        C_vec: numpy.ndarray
            An array of active nodes. Where the column represents
            the nodes. The rows represents the data points.

        Returns:
        --------
        theta_sum: numpy.ndarray
            An array of theta_sum for C_vec.
        """
        C_vec = np.array(C_vec, dtype=bool)
        C_vec = np.atleast_2d(C_vec)
        theta_sum = np.zeros(C_vec.shape[0])
        for C, theta in zip(self.C_vec, self.theta_vec):
            idx = np.sum(C_vec & C, axis=1) == np.sum(C)
            theta_sum[idx] += theta
        return theta_sum

    def theta2p(self, C_vec):
        """
        Returns the un-normalised probability of C_vec.

        Parameters:
        -----------
        C_vec: numpy.ndarray
            An array of active nodes. Where the column represents
            the nodes. The rows represents the data points.

        Returns:
        --------
        p_vec: numpy.ndarray
            Probability vector of C_vec
        """
        theta_sum = self.get_theta_sum(C_vec)
        p_vec = np.exp(theta_sum)
        return p_vec

    def get_model_predictions_by_gibbs_sample(self, N_gibbs_samples=100000, burn_in=3):
        """
        Compute P using Gibbs sampling. This should not be used when the feature space
        and label space is large.

        Parameters:
        -----------
        N_gibbs_samples: int
            Number of data points used for Gibbs sampling
        burn_in: int
            Number of samples before using the data point

        Returns:
        --------
        unique_X: numpy.ndarray
            An array of active nodes. Where the column represents
            the nodes. The rows represents the data points.
        p_vec: numpy.ndarray
            The vector of probability for unique_X approximated by
            Gibbs sampling.
        """
        X_prediction = self._get_model_samples(N_gibbs_samples=N_gibbs_samples, burn_in=burn_in)
        unique_X, count_X = np.unique(X_prediction, return_counts=True, axis=0)
        sum_count_X = np.sum(count_X)
        p_vec = np.array([count / sum_count_X for C, count in zip(unique_X, count_X)])
        return unique_X, p_vec

    def get_emperical_p(self):
        """
        Get the probability distribution of the input.

        Parameters:
        -----------
        None

        Returns:
        --------
        C_vec: numpy.ndarray
            An array of active nodes. Where the column represents
            the nodes. The rows represents the data points.
        p_vec: numpy.ndarray
            Probability vector of C_vec
        """
        p_vec = list()
        C_vec = list()
        for C in np.unique(self.X, axis=0):
            idx = np.sum(self.X == C, axis=1) == self.X.shape[1]
            p = np.sum(idx) / len(idx)
            p_vec.append(p)
            C_vec.append(C)
        p_vec = np.array(p_vec)
        C_vec = np.array(C_vec)
        return C_vec, p_vec

    def _ais_fj(self, C_vec, log_f_n):
        """
        Returns the intermediate probability distribution. Defined
        by p_k (x) prop p_A^* (x) ^(1-beta_k) p_B^* (x) ^(beta_k).
        Where p_A is the base distribution and p_B is the current
        proposed distribution by the model.

        Parameters:
        -----------
        C_vec: numpy.ndarray
            An array of active nodes. Where the column represents
            the nodes. The rows represents the data points.

        Returns:
        --------
        log_f_j: numpy.ndarray
            The intermediate probability for C_vec.            
        """
        log_f_j = ((1 - self.ais_beta[self.ais_beta_idx]) * log_f_n) \
                + (self.ais_beta[self.ais_beta_idx] * self.ais_log_f_0)
        return log_f_j

    def _ais_step(self, C_vec):
        """
        Parameters:
        -----------
        C_vec: numpy.ndarray
            An array of active nodes. Where the column represents
            the nodes. The rows represents the data points.
        """
        log_f_n = self.get_theta_sum(C_vec)
        self.ais_log_w -= self._ais_fj(C_vec, log_f_n)
        self.ais_beta_idx += 1
        self.ais_log_w += self._ais_fj(C_vec, log_f_n)

    def _ais_init(self, beta, n_samples=10000):
        """
        This function initialises the parameters for Annealed Important Sampling (AIS).

        Parameters:
        -----------
        n_samples: int
            The number of Gibbs samples.

        Returns:
        --------
        None
        """
        self.ais_n_samples = n_samples
        self.ais_log_w = np.full(n_samples, 0, dtype=float)
        self.ais_log_f_0 = np.full(n_samples, -self.log_N_max, dtype=float)
        self.ais_beta = beta
        self.ais_beta_idx = 0
        self.log_Z_0 = self.theta_vec[0]

    def _ais_evaluate(self):
        """
        This function approximate the value for log(Z) by using AIS.

        Parameters:
        -----------
        None

        Returns:
        --------
        log_Z_n: float
            The expected value for log(Z) approximated by AIS.
        log_Z_n_3std_lower: float
            The lower bound for log(Z) approximated by AIS.
        log_Z_n_3std_upper: float
            The upper bound for log(Z) approximated by AIS.
        """
        log_r_ais = np.log(np.mean(np.exp(self.ais_log_w)))
        log_Z_n = log_r_ais + self.log_Z_0

        mu = np.exp(log_r_ais)
        log_Z_n_std = np.log(np.std(np.exp(self.ais_log_w) - mu) + mu) - (np.log(self.ais_n_samples) / 2)
        log_Z_n_3std = np.log(3) + log_Z_n_std
        log_Z_n_3std_upper = np.log(np.exp(log_r_ais) + np.exp(log_Z_n_3std)) + self.log_Z_0
        log_Z_n_3std_lower = np.log(np.exp(log_r_ais) - np.exp(log_Z_n_3std)) + self.log_Z_0

        return log_Z_n, log_Z_n_3std_lower, log_Z_n_3std_upper

    def ais2p(self, C_vec):
        """
        Compute the upper and lower bound for the probability distribution

        Parameters:
        -----------
        C_vec: numpy.ndarray
            An array of active nodes. Where the column represents
            the nodes. The rows represents the data points.

        Returns:
        --------
        p_vec: numpy.ndarray
            The expected probability approximated by AIS
        p_lower_vec: numpy.ndarray
            The lower bound approximated by AIS. The lower bound is
            approximated by 3 standard deviations.
        p_upper_vec: numpy.ndarray
            The upper bound approximated by AIS. The upper bound is
            approximated by 3 standard deviations.
        """
        C_vec = np.array(C_vec, dtype=bool)
        log_Z_n, log_Z_n_3std_down, log_Z_n_3std_up = self._ais_evaluate()

        theta_prev = self.theta_vec[0]

        self.theta_vec[0] = log_Z_n_3std_down
        p_lower_vec = self.theta2p(C_vec)

        self.theta_vec[0] = log_Z_n_3std_up
        p_upper_vec = self.theta2p(C_vec)

        self.theta_vec[0] = log_Z_n
        p_vec = self.theta2p(C_vec)

        self.theta_vec[0] = theta_prev

        return p_vec, p_lower_vec, p_upper_vec

    def save(self, file_name):
        """
        Save checkpoint.
        """
        fid = open(file_name, 'wb')
        pickle.dump(self.__dict__, fid)
        fid.close()


    def load(self, file_name):
        """
        Load checkpoint.
        """
        fid = open(file_name, 'rb')
        tmp_dict = pickle.load(fid)
        fid.close()

        self.__dict__.update(tmp_dict) 
