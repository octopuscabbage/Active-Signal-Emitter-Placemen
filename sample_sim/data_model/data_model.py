import abc
import pickle

import numpy as np

from sample_sim.data_model.gp_wrapper import TorchSparseUncertainGPModel, TorchExactGp, GPWrapper
from sample_sim.general_utils import is_good_matrix


class DataModel(abc.ABC):
    def __init__(self, logger, verbose=False):
        self.Xs = None
        self.Ys = None
        self.verbose = verbose
        self.logger = logger

    def update(self, X, Y):
        if self.Xs is None:
            self.Xs = X
        else:
            self.Xs = np.vstack((self.Xs, X))
        if self.Ys is None:
            self.Ys = Y
        else:
            self.Ys = np.append(self.Ys,Y)

    def query(self, p, return_std=True):
        return self.query_many(np.array([p]), return_std=return_std)

    def query_many(self, Xs, return_std=True):
        if return_std:
            mean, std = self.query_many_implementation__(Xs, return_std)
        else:
            std = None
            mean = self.query_many_implementation__(Xs, return_std)
        if return_std:
            return mean, std
        else:
            return mean

    @abc.abstractmethod
    def query_many_implementation__(self, Xs, return_std=True):
        pass

    def _flatten_data(self):
        if isinstance(self.Xs, list):
            self.Xs = np.vstack(self.Xs)
            self.Ys = np.concatenate(self.Ys)
            assert is_good_matrix(self.Xs)
            assert is_good_matrix(self.Ys)

    @abc.abstractmethod
    def update_prior(self,X,Y):
        pass

class TorchDataModel(DataModel):
    def __init__(self, logger, model: GPWrapper):
        super().__init__(logger,verbose=True)
        self.logger = logger
        self.model = model

    def update(self, X, Y):
        super().update(X, Y)
        self.model.update_prior(self.Xs,self.Ys)

    def fit(self,steps=200):
        self.model.fit(self.Xs,self.Ys,optimization_steps=steps)

    def query_many_implementation__(self, Xs, return_std=True):
        return self.model.predict(Xs, return_std)

    def save(self, fname):
        with open(fname + "dm.pkl", "wb") as f:
            pickle.dump(self.Xs, f)
            pickle.dump(self.Ys, f)
            pickle.dump(self.input_uncertanties, f)
            pickle.dump(self.use_uncertainty, f)
        self.model.save(fname)

    def load(self, fname):
        with open(fname + "dm.pkl", "rb") as f:
            self.Xs = pickle.load(f)
            self.Ys = pickle.load(f)
            self.input_uncertanties = pickle.load(f)
            self.use_uncertainty = pickle.load(f)
        self.model.load(fname)
    def update_prior(self, X, Y):
        return self.model.update_prior(X,Y)
    def clear_prior(self):
        self.model.update_prior(self.Xs,self.Ys)

class TorchApproximateGPBackedDataModel(TorchDataModel):
    def __init__(self, logger, inducing_points=None, verbose=False, use_x_as_inducing=True):
        self.refit = True
        self.gp = TorchSparseUncertainGPModel(logger, inducing_points, use_fast_strategy=False)
        self.gp.verbose = verbose
        self.use_x_as_inducing = use_x_as_inducing

        super().__init__(logger, model=self.gp)


class TorchExactGPBackedDataModel(TorchDataModel):
    def __init__(self, X, Y, logger,use_better_mean=False,force_cpu=False,device=None):
        self.gp = TorchExactGp(X, Y, logger=logger, use_mlp_mean=use_better_mean,force_cpu=force_cpu,gpu_num=device)
        super().__init__(logger,model=self.gp)

