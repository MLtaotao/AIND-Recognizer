import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    logL: log Likelihood
    N: X.shape[0]
    Free parameters(p):
    1. The free transition probability parameters, which is the size of the transmat matrix less one row because they add up to 1 and therefore the final row is deterministic, so `n*(n-1)`
    2. The free starting probabilities, which is the size of startprob minus 1 because it adds to 1.0 and last one can be calculated so `n-1`
    3. Number of means, which is `n*f`
    4. Number of covariances which is the size of the covars matrix, which for "diag" is `n*f`
    sum of them so
    p = n**2 + 2*n*f - 1
    References:
    [1]https://en.wikipedia.org/wiki/Hidden_Markov_model
    [2]https://stats.stackexchange.com/questions/12341/number-of-parameters-in-markov-model
    [3]https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/8
    """
    def Compute_Bic(self, n):
        logL = self.base_model(n).score(self.X, self.lengths)
        p = n**2 + 2 * sum(self.lengths) * n -1
        return -2 * logL + p * np.log(self.X.shape[0]), self.base_model(n)

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        models_bic = {}

        for states_num in range(self.min_n_components, self.max_n_components + 1):
            current_bic, current_model = float("inf"), None
            try:
                current_bic, current_model = self.Compute_Bic(states_num)
            except:
                pass
            models_bic[current_model] = current_bic

        return min(models_bic, key = models_bic.get)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)*SUM(log(P(X(all but i))

    log(P(X(i)):likelihood of the data
    log(P(X(all but i): anti log likelihood of data X except i
    '''
    def Compute_Dic(self, states_num, other_word):
        logL = self.base_model(states_num).score(self.X, self.lengths)
        logL_all_but_i = [self.base_model(states_num).score(word[0], word[1]) for word in other_word]
        return logL - np.mean(logL_all_but_i), self.base_model(states_num)

    def select(self):
        # TODO implement model selection based on DIC scores
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        other_word = [self.hwords[w] for w in self.words if w != self.this_word]

        models_dic = {}
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            current_dic, current_model = float("-inf"), None

            try:
                current_dic, current_model = self.Compute_Dic(num_states, other_word)

            except:
                pass
            models_dic[current_model] = current_dic

        return max(models_dic, key = models_dic.get)



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def logl_cv(self, num_states):

        logls = []
        if len(self.sequences) > 2:
            split_method = KFold()
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)

                logls.append(self.base_model(num_states).score(test_X, test_lengths))
        else:
            self.X, self.lengths = self.hwords[self.this_word]
            logls.append(self.base_model(num_states).score(self.X, self.lengths))
        return np.mean(logls), self.base_model(num_states)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        models_cvlogl = {}

        for num_states in range(self.min_n_components, self.max_n_components+1):

            current_cv_logl, current_model = float("-inf"), None
            try:
                current_cv_logl, current_model = self.logl_cv(num_states)

            except:
                pass

            models_cvlogl[current_model] = current_cv_logl

        return max(models_cvlogl, key = models_cvlogl.get)
