# Author：JohnMa
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import is_regressor
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import check_random_state

class ImAdaBoostClassifier(AdaBoostClassifier):
    def __init__(self, base_estimator=None, n_estimators=150, learning_rate=0.5,num =1,
                 algorithm='SAMME.R', random_state=None):
        super().__init__(
            base_estimator=base_estimator,n_estimators=n_estimators,
            learning_rate=learning_rate, random_state=random_state,algorithm=algorithm)
        self.num = num
        self.algorithm = algorithm

    def fit(self, X, y, sample_weight=None):
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        X, y = self._validate_data(X, y,
                                   accept_sparse=['csr', 'csc'],
                                   ensure_2d=True,
                                   allow_nd=True,
                                   dtype=None,
                                   y_numeric=is_regressor(self))
        sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        sample_weight /= sample_weight.sum()
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")
     
        self._validate_estimator()
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64) 
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)
        
        for iboost in range(self.n_estimators):
            sample_weight, estimator_weight, estimator_error = self._boost_imada(
                iboost,
                X, y,
                sample_weight,
                random_state)

            if (estimator_error is None or estimator_error >0.5):
                continue

            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                sample_weight /= sample_weight_sum
        return self

    def _boost_imada(self, iboost, X, y, sample_weight, random_state):
        if self.algorithm == 'SAMME.R':
            return self._boost_real_imada(iboost, X, y, sample_weight, random_state)

        else:
            return self._boost_discrete_imada(iboost, X, y, sample_weight, random_state)

    def _boost_discrete_imada(self, iboost, X, y, sample_weight, random_state):
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        incorrect = y_predict != y

        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in ImAdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        estimator_weight = self.learning_rate * (
                np.log((1. - estimator_error) / estimator_error) +
                np.log(n_classes - 1.))

        if not iboost == self.n_estimators - 1:
            incorrect = np.array(list(map(lambda x: 1 if x==True else -1,incorrect)))
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    (sample_weight > 0)*(self._beta(y, y_predict,iboost,self.num)))

        return sample_weight, estimator_weight, estimator_error

    def _boost_real_imada(self, iboost, X, y, sample_weight, random_state):
        estimator = self._make_estimator(random_state=random_state) #base
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)
        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)
        incorrect = y_predict != y
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])
        proba = y_predict_proba  # alias for readability

        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
        
        estimator_weight = (-1. * self.learning_rate * ((n_classes - 1.) / n_classes)
                            * (y_coding * np.log(y_predict_proba)).sum(axis=1))

        if not iboost == self.n_estimators - 1:
            a = sample_weight
            criteria = ((sample_weight > 0) | (estimator_weight < 0))
            sample_weight *= np.exp(estimator_weight * criteria * self._beta(y, y_predict,iboost,self.num))
        return sample_weight, 1., estimator_error
    
    def _beta(self, y, y_hat,iboost,num):
        res = []
        x1 = np.sum((y != y_hat))
        x2 = np.sum((y != y_hat) & (y==1))
        num1 = x2/x1
        num0 = 1-num1
        for m,i in enumerate(zip(y, y_hat)):
            if i[0]==i[1]:
                res.append(1.5*num)
            elif i[0] != i[1]:
                if i[0]==1 and i[1]==0 and num1!=num0:
                    if num1<num0:
                        res.append((-0.5 * num1 + 2)*num)
                    else:
                        res.append((-0.5 * num1 + 1.5)*num)
                elif i[0]==0 and i[1]==1 and num1!=num0:
                    if num1<num0:
                        res.append((0.5 * num1+1)*num)
                    else:
                        res.append((0.5 * num1 + 1.5)*num)
                elif num1==num0:
                    res.append(1.5*num)
        return np.array(res)
