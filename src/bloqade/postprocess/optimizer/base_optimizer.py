# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:09:25 2023

@author: jwurtz
"""
from typing import Callable, Dict, List, Tuple, Union
from scipy.optimize import minimize
from skopt import gp_minimize
import time
import json
import numpy as np

import bloqade.postprocess as postprocess
import bloqade

import functools

from braket.jobs import (
    InstanceConfig,
    hybrid_job,
)


# %%
class Optimizer:
    """
    The `Optimizer` class provides a comprehensive framework for optimizing
     functions as well as storing results to and from file.

    Methods:
    - callback : Callback function when executing the optimizer
    - from_file: Load previous results from file
    - to_file  : Dump results to file
    - Optimize : Optimize the function given pre-set configuration
    - call     : Optimizer() : Same functionality as Optimize.
    """

    def __init__(
        self,
        ansatz: Callable,  # Anything that is a callable should work! This is generic.
        method: str,
        initial_guess: list,
        max_iter: int,
        recorder: postprocess.record = None,
        delta: float = 0.01,
        learning_rate: float = 0.005,
        bounds: List[Tuple[float, float]] = None,
        fixed_indices: List[int] = [],
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        ansatz : Callable
            The function to be minimized. Must be a callable. Note that while
             the object is designed to optimize variational quantum circuits,
             the ansatz can be any callable, including purely classical functions.
        method : str
            The method to do optimization. Can be:
                "SPSA" - SPSA optimization, which is a randomized gradient descent
                "Bayesian" - Bayesian optimization
                other str  - An optimization method for scipy.optimize.minimize
        initial_guess : list
            The initial guess to provide to the optimizer.
        max_iter : int
            Maximum number of iteration steps for the optimizer
        recorder : postprocess.record, optional. Default is None
            A postprocess.record object, which manages file IO and stores data
            and results to file
        delta : float, optional. The default is 0.01.
            The dx size for which to compute the discrete derivative.
            Required for SPSA optimization.
        learning_rate : float, optional. The default is 0.005.
            The dx size to step for SPSA optimization.
        bounds : List[Tuple[float, float]], optional. The default is None.
            The parameter bounds for optimization. Required for Bayesian optimization.
            For more details, see the documentation of skopt.gp_minimize.
        fixed_indices : indices of parameters for which to keep fixed and not optimize.
        verbose : bool, optional. The default is False.
            True - Print status after every iteration step
            False- Do not print anything to console
        """

        self.method = method
        self.tracker = []
        self.niters = 0
        self.max_iter = max_iter
        self.verbose = verbose
        if recorder is None:
            self.recorder = postprocess.record.null_record()
        else:
            self.recorder = recorder

        self.delta = delta
        self.learning_rate = learning_rate

        self.bounds = bounds

        self.ansatz = ansatz

        self.fixed_indices = fixed_indices

        self.D = len(initial_guess)
        self.free_indices = list(set(range(self.D)).difference(fixed_indices))
        self.free_indices.sort()
        self.x_init, self.fixed_values = self._unflatten_parameters(initial_guess)

    def callback(self, state):
        payload = [self._flatten_parameters(list(state["x"])), state["fun"]]
        self.tracker.append(payload)

        self.recorder.put(self)

        self.niters += 1
        if self.verbose == True:
            print(self.niters, payload)

    def _flatten_parameters(self, x):
        assert len(x) == len(self.free_indices)
        xin = [None] * self.D
        for k, ind in enumerate(self.free_indices):
            xin[ind] = x[k]
        for k, ind in enumerate(self.fixed_indices):
            xin[ind] = self.fixed_values[k]
        return xin

    def _unflatten_parameters(self, x):
        return (
            [x[ind] for ind in self.free_indices],
            [x[ind] for ind in self.fixed_indices],
        )

    def function(self, x):
        return self.ansatz(self._flatten_parameters(x))

    def __call__(self, fixed_vals=[]):
        return self.optimize(fixed_vals)

    def __str__(self):
        payload = {}
        payload["method"] = self.method
        payload["tracker"] = self.tracker
        payload["niters"] = self.niters
        payload["max_iter"] = self.max_iter
        payload["verbose"] = self.verbose
        payload["delta"] = self.delta
        payload["learning_rate"] = self.learning_rate
        payload["x_init"] = self.x_init
        payload["bounds"] = self.bounds
        return json.dumps(payload)

    @classmethod
    def from_file(cls, ansatz, recorder: postprocess.record.record):
        payload = json.loads(recorder.get(cls))
        dout = Optimizer(
            ansatz=ansatz,
            method=payload["method"],
            initial_guess=payload["x_init"],
            max_iter=payload["max_iter"],
            recorder=recorder,
            delta=payload["delta"],
            learning_rate=payload["learning_rate"],
            bounds=payload["bounds"],
            verbose=payload["verbose"],
        )

        dout.tracker = payload["tracker"]

        return dout

    def to_file(self, recorder: postprocess.record.record):
        recorder.put(self)

    def _spsa(
        self, delta: float, learning_rate: float
    ) -> Dict[str, Union[float, List[float], np.ndarray, str, bool]]:
        """
        Implements the Simultaneous Perturbation Stochastic Approximation (SPSA) optimization algorithm.

        SPSA is a gradient-free optimization method suitable for noisy cost functions. The method approximates
        the gradient using simultaneous random perturbations, which makes it especially useful for quantum
        optimization problems.
        """

        # Initialize variables
        N = len(self.x_init)
        x = self.x_init.copy()

        for _ in range(self.max_iter):
            rand_vec = 2 * np.random.binomial(1, 0.5, N) - 1
            x_plus = x + delta * rand_vec
            x_minus = x - delta * rand_vec

            y_plus = self.function(x_plus)
            y_minus = self.function(x_minus)

            g_hat = (y_plus - y_minus) / (2.0 * delta * rand_vec)
            x -= learning_rate * g_hat

            status = {"fun": self.function(x), "x": x}
            self.callback(status)

        # Generate the same output as scipy.optimize.minimize
        return {
            "fun": self.function(x),
            "x": x,
            "success": True,
            "message": "",
            "nit": self.max_iter,
        }

    def _random(self):
        """
        Randomly pick parameters uniformly from the range
        """

        N = len(self.x_init)
        bounds = np.array(self.bounds)
        width = np.diff(bounds, 1)[:, 0]

        best_y = 1e99
        best_x = None

        for _ in range(self.max_iter):
            x = np.random.rand(N) * width + bounds[:, 0]
            y = self.function(x.tolist())
            status = {"fun": y, "x": x.tolist()}
            self.callback(status)

            if y < best_y:
                best_y = y
                best_x = x

        return {
            "fun": best_y,
            "x": best_x,
            "success": True,
            "message": "",
            "nit": self.max_iter,
        }

    def _bayesian(self):
        """
        Implements the Bayesian optimization method using Gaussian Processes.

        Bayesian optimization is a model-based optimization method that uses a probabilistic model
        (typically Gaussian Processes) to predict the function values and decide where to evaluate next.
        It's particularly useful for optimizing expensive black-box functions.
        """

        result = gp_minimize(
            self.function,
            self.bounds,
            x0=list(self.x_init),
            n_calls=self.max_iter,
            callback=self.callback,
        )

        return result

    def optimize(self, fixed_values=[]) -> Dict:
        """
        Main method to perform optimization based on the chosen method.

        The function internally calls the appropriate optimization routine (e.g., SPSA, Bayesian, etc.)
        and returns the optimization result. The result typically includes the best parameters found,
        the value of the cost function at those parameters, and other relevant information.
        """

        self.fixed_values = fixed_values

        if self.method == "SPSA":
            result = self._spsa(self.delta, self.learning_rate)
        elif self.method == "Bayesian":
            result = self._bayesian()
        elif self.method == "Random":
            result = self._random()
        else:
            print("Unknow method, resorting to SKOpt")
            result = minimize(
                self.function,
                self.x_init,
                method=self.method,
                options={"maxiter": self.max_iter},
                callback=self.callback,
            )
        self.result = result
        self.recorder.put(self)

        self.x_init = list(result["x"])
        return result["fun"]

    def remote_call(
        self, device: str = None, local: bool = False, job_name: str = None
    ):
        """
        Sets the job onto AWS BraKet using a particular device. Not implemented!
        """
        raise NotImplementedError("I have not yet figured out how to do remote hybrid!")
        # Generate the hybrid job instance (I'm doing it this way instead of the decorator)
        hj = hybrid_job(
            device=device,
            local=local,
            job_name=job_name,
            include_modules=[bloqade, json, postprocess],
        )
        # Do it this way instead of using a decorator @hyrbid_job
        remote = hj(self)

        return remote()
