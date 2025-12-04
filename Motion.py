"""
This is an aggregate class where a user can instantiate a motion with the specified simulator to simulate
the motion, and then a specified level of freed parameters within the motion
"""

import os

from torchgen.api.cpp import return_names

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bayesflow as bf
import keras
import os
import jax


class Motion:
    def __init__(self, simulator : str, parameters : str, with_prior : bool = True,
                 seed : int = int(os.times()[4]),
                 time : float = 100 / 365, num_steps : int = 100,
                 x0 : list = [100, 100, 100],
                 drift_scale = 0.4,
                 vol_mean = 0.24621856131518247, vol_stdev = 0.0049087692859631936 * 100,
                 corr_max = 1):
        """
        Simulation -- "geom":GBM; "arith":ABM
        Parameters -- "fc":full covariance; "diag":diagonal covariance; "sph":spherical covariance

        Usage:
        First, create an instance of the Motion class with the simulator and parameters you want to simulate.
        --> my_motion = Motion(simulator="geom", parameters="fc")

        Then, call the Motion object you created as a functor to simulate a motion with the given simulator
        and parmeters
        --> path = my_motion()

        This returns a dictionary where the "motion" key corresponds to the simulated path, which is
        a list of 3-element lists, where each super-list is a time step, and each element of the 3-element
        sub-list is one of the three correlated (assuming full covariance) motions.
        """

        self.simulator = simulator.lower()
        self.parameters = parameters.lower()
        self.with_prior = with_prior
        self.RNG = np.random.default_rng(seed)
        self.time = time
        self.num_steps = num_steps
        self.x0 = x0
        self.drift_scale = drift_scale
        self.vol_mean = vol_mean
        self.vol_stdev = vol_stdev
        self.corr_max = corr_max

    def __drift_prior(self):
        # Generates a random draw from the prior
        b1, b2, b3 = self.RNG.uniform(-self.drift_scale, self.drift_scale, size=3)
        #b1, b2, b3 = 0.2, 0.4, -0.3

        return {"b1": b1, "b2": b2, "b3": b3}

    def __vol_prior(self, spherical=False):
        if not spherical:
            vols = {"v1":self.RNG.lognormal(mean=self.vol_mean, sigma=self.vol_stdev),
                    "v2":self.RNG.lognormal(mean=self.vol_mean, sigma=self.vol_stdev),
                    "v3":self.RNG.lognormal(mean=self.vol_mean, sigma=self.vol_stdev)}
        else:
            vol = self.RNG.lognormal(mean=self.vol_mean, sigma=self.vol_stdev)
            vols = {"v1":vol, "v2":vol, "v3":vol}

        return vols

    def __corr_prior(self):
        max_tries = 100
        i = 0

        # Draw uniformly from a cube and check if it is a valid correlation matrix
        while i < max_tries:
            m21, m31, m32 = self.RNG.uniform(0, self.corr_max, size=3)

            if (m21**2 + m31**2 + m32**2 - 2*m21*m31*m32) <= 1:
                break

            i += 1

        if (i >= max_tries):
            raise ValueError("Could not generate valid correlation matrix after {} tries".format(max_tries))

        return {"m21": m21, "m31": m31, "m32": m32}

    def __GBM_sim(self, b1 = None, b2 = None, b3 = None,
                v1 = None, v2 = None, v3 = None,
                m21 = None, m31 = None, m32 = None):

        dt = self.time / self.num_steps

        stdevs = np.array([v1, v2, v3])
        stdevs_D = np.diag(stdevs)
        correlation = np.array([[1.0, m21, m31],
                                [m21, 1.0, m32],
                                [m31, m32, 1.0]])
        sigma = correlation * np.outer(stdevs, stdevs)

        L = np.linalg.cholesky(sigma)

        x = self.x0

        motion = [self.x0]

        drift_coef = np.array([b1, b2, b3])
        correction = 0.5 * np.sum(L ** 2, axis=1)
        drift = (drift_coef - correction) * dt

        for _ in range(0, self.num_steps - 1):
            random_shocks = L @ self.RNG.normal(scale=np.sqrt(dt), size=3)
            x = x * np.exp(drift + random_shocks)
            motion.append(x)

        return dict(motion=np.asarray(motion))

    def __ABM_sim(self, b1=None, b2=None, b3=None,
                v1=None, v2=None, v3=None,
                m21=None, m31=None, m32=None):

        dt = self.time / self.num_steps

        stdevs = np.array([v1, v2, v3])
        stdevs_D = np.diag(stdevs)
        correlation = np.array([[1.0, m21, m31],
                                [m21, 1.0, m32],
                                [m31, m32, 1.0]])
        sigma = correlation * np.outer(stdevs, stdevs)

        L = np.linalg.cholesky(sigma)

        x = self.x0

        motion = [self.x0]

        drift = np.array([b1, b2, b3]) * dt
        correction = 0.5 * np.sum(L ** 2, axis=1)

        for _ in range(0, self.num_steps - 1):
            random_shocks = L @ self.RNG.normal(scale=np.sqrt(dt), size=3) * np.sqrt(dt)
            dx = drift + random_shocks
            x = x + dx
            motion.append(x)

        return dict(motion=np.asarray(motion))

    def __call__(self):
        if self.simulator == "geom":
            sim = self.__GBM_sim
        elif self.simulator == "arith":
            sim = self.__ABM_sim
        else:
            raise ValueError("Simulator must be either 'geom' or 'arith'")

        drifts = self.__drift_prior()

        volatilities = dict()
        correlations = dict()

        if self.parameters == "fc":
            volatilities = self.__vol_prior()
            correlations = self.__corr_prior()
        elif self.parameters == "diag":
            volatilities = self.__vol_prior()
            correlations = {"m21": 0, "m31": 0, "m32": 0}
        elif self.parameters == "sph":
            volatilities = self.__vol_prior(spherical=True)
            correlations = {"m21": 0, "m31": 0, "m32": 0}
        else:
            raise ValueError("Parameters must be either 'fc', 'diag', or 'sph'")

        motion = sim(**drifts, **volatilities, **correlations)

        if self.with_prior:
            return motion | drifts | volatilities | correlations
        else:
            return motion

if __name__ == "__main__":
    days = 100
    time = days/365
    steps = days

    ppc_sims = dict()

    fc_simulator = Motion("geom", "fc", time=time, num_steps=steps)
    ppc_sims["fc"] = [fc_simulator()["motion"] for _ in range(1000)]

    diag_simulator = Motion("geom", "diag", time=time, num_steps=steps)
    ppc_sims["diag"] = [diag_simulator()["motion"] for _ in range(1000)]

    sph_simulator = Motion("geom", "sph", time=time, num_steps=steps)
    ppc_sims["sph"] = [sph_simulator()["motion"] for _ in range(1000)]

    dims = 3
    fig, axes = plt.subplots(dims, dims, figsize=(8, 8))
    fig.suptitle("Prior Predictive Check")

    for i in range(3):
        for j in range(dims):
            x = range(0, steps)

            ax = axes[i, j]
            if i == 0:
                sims = ppc_sims["fc"]
            elif i == 1:
                sims = ppc_sims["diag"]
            elif i == 2:
                sims = ppc_sims["sph"]
            else:
                raise Exception("Sim out of range")

            variable = [[step[j] for step in sim] for sim in sims]

            ax.set_facecolor("white")

            num_sample_paths = 20
            for line_idx in range(num_sample_paths):
                ax.plot(x, variable[line_idx], color="lightblue")

            mean_line_start = (0, variable[0][0])
            mean_line_end = (steps, np.mean([sim[-1] for sim in variable]))
            ax.axline(mean_line_start, mean_line_end, color="r")

            slope = (mean_line_end[1] - mean_line_start[1]) / (mean_line_end[0] - mean_line_start[0])
            ax.annotate(f"{slope:.3f}", mean_line_end)

            xlab = ""
            ylab = ""
            if j == 0:
                if i == 0:
                    ylab = "Full Covariance"
                elif i == 1:
                    ylab = "Diagonal Covariance"
                elif i == 2:
                    ylab = "Spherical Covariance"

            if i == 2:
                if j == 0:
                    xlab = "Motion 1"
                elif j == 1:
                    xlab = "Motion 2"
                elif j == 2:
                    xlab = "Motion 3"

            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)

    plt.tight_layout()

    plt.show()