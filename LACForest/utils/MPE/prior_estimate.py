from utils.MPE.kmp import wrapper as kmp_wrapper

import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC

# Basic class for various Estimators


def squared_dist(x, c):
    #    assert x.shape[1] == c.shape[1], "Dimension must be the same."
    return np.sum(x**2, axis=1, keepdims=True) \
        + np.sum(c**2, axis=1, keepdims=True).T \
        - 2 * x.dot(c.T)


class PriorEstimator(object):
    def __init__(self, prior_given=0.7):
        self.prior_given = prior_given
        self.wrapper = None

    def estimate(self, X_train, X_unlabel):
        raise NotImplementedError


class KernelPriorEstimator(PriorEstimator):
    def __init__(self, prior_given=0.7):
        super(KernelPriorEstimator, self).__init__(prior_given)
        self.wrapper = kmp_wrapper
        self.n_fold = 5

    def estimate(self, X_train, X_unlabel):
        X = np.concatenate((X_train, X_unlabel), axis=0)
        y = np.concatenate((np.ones(len(X_train)),
                            -1 * np.ones(len(X_unlabel))), axis=0)

        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=2020)
        score = np.zeros(len(X), np.float64)

        for train_index, test_index in kf.split(X):
            X_tr, X_test = X[train_index], X[test_index]
            y_tr, y_test = y[train_index], y[test_index]
            # dp = distance_matrix(X_tr, X_tr, p=2) ** 2
            dp = squared_dist(X_tr, X_tr)
            gamma = 1 / np.median(dp)
            clf = SVC(kernel='rbf', gamma=gamma)
            clf.fit(X_tr, y_tr)
            score[test_index] = clf.decision_function(X_test)

        score_train = score[:len(X_train)]
        score_unlabel = score[len(X_train):]

        prior_estimate = self.wrapper(
            score_unlabel, score_train, len(X_unlabel), len(X_train), 1)

        return prior_estimate[1]
