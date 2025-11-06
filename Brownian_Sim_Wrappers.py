"""
Defines wrapper classes to combine the prior function and the simulator into a single class
which outputs both the prior and the simulated motion on call
"""

import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import bayesflow as bf
import os
import jax


# GBM volatiilty-based simulation wrapper
class GBMVolWrapper:
    def __init__(self, seed : int = int(os.times()[4]), mean = 0.24621856131518247, stdev = 0.0049087692859631936*100):
        self.RNG = np.random.default_rng()
        self.mean = mean
        self.stdev = stdev

    def prior(self):
        # Generates a random draw from the prior

        v1 = self.RNG.lognormal(mean=self.mean, sigma=self.stdev)
        v2 = self.RNG.lognormal(mean=self.mean, sigma=self.stdev)
        v3 = self.RNG.lognormal(mean=self.mean, sigma=self.stdev)

        return {"v1": v1, "v2": v2, "v3": v3}

    def GBM_sim(self, v1, v2, v3, x0=np.array([100, 100, 100]), time=100 / 365, time_step=1 / 365):
        stdevs = np.array([v1, v2, v3])
        stdevs_D = np.diag(stdevs)
        correlation = np.array([[1.0, 0.4472136, 0.0],
                                [0.0, 1.0, 2.12132],
                                [0.0, 0.0, 1.0]])
        sigma = np.dot(stdevs_D, correlation, stdevs_D)

        b1, b2, b3 = 0.2, 0.4, -0.3

        x = x0

        motion = [x0]

        for _ in range(0, int(time / time_step) - 1):
            drift_coef = np.array([b1, b2, b3])
            correction = 0.5 * np.sum([sigma[:, j] ** 2 for j in range(0, 3)], axis=0)
            drift = drift_coef - correction
            timescaled_drift = drift * time_step
            random_shocks = sigma @ self.RNG.normal(scale=np.sqrt(time_step), size=3)
            dx = x * (timescaled_drift + random_shocks)
            x = x + dx
            motion.append(x)

        return dict(motion=np.asarray(motion))

    def __call__(self):
        drawn_prior = self.prior()
        motion = self.GBM_sim(**drawn_prior)

        return drawn_prior | motion


# GBM correlation-based simulation wrapper
class GBMCorrWrapper:
    def __init__(self, seed : int = int(os.times()[4]), corr_scale = 1):
        self.RNG = np.random.default_rng()
        self.scale = corr_scale

    def prior(self):
        rho12 = self.RNG.uniform(-self.scale, self.scale)
        rho13 = self.RNG.uniform(-self.scale, self.scale)
        rho23 = self.RNG.uniform(-self.scale, self.scale)

        return {"rho12": rho12, "rho13": rho13, "rho23": rho23}

    def GBM_sim(self, rho12, rho13, rho23, x0=np.array([100, 100, 100]), time=100 / 365, time_step=1 / 365):
        # Fixed volatilities to isolate correlation learning
        vols = np.array([0.25, 0.25, 0.25])
        D = np.diag(vols)

        # build correlation matrix from parameters
        corr = np.array([
            [1.0, rho12, rho13],
            [rho12, 1.0, rho23],
            [rho13, rho23, 1.0]
        ])

        # covariance
        sigma = D @ corr @ D

        b = np.array([0.2, 0.4, -0.3])
        x = x0.copy()
        motion = [x0]

        for _ in range(int(time / time_step) - 1):
            correction = 0.5 * np.sum([sigma[:, j] ** 2 for j in range(3)], axis=0)
            drift = b - correction
            drift_term = drift * time_step
            shocks = sigma @ self.RNG.normal(scale=np.sqrt(time_step), size=3)
            dx = x * (drift_term + shocks)
            x = x + dx
            motion.append(x)

        return dict(motion=np.asarray(motion))

    def __call__(self):
        drawn_prior = self.prior()
        motion = self.GBM_sim(**drawn_prior)

        return drawn_prior | motion


# GBM covariance matrix-based simulation wrapper
class GBMCovWrapper:
    def __init__(self, seed : int = int(os.times()[4]), mean = 0.24621856131518247, stdev = 0.0049087692859631936*100):
        self.RNG = np.random.default_rng()
        self.mean = mean
        self.stdev = stdev

    def prior(self):
        # Generates a random draw from the prior

        v1 = self.RNG.lognormal(mean=self.mean, sigma=self.stdev)
        v2 = self.RNG.lognormal(mean=self.mean, sigma=self.stdev)
        v3 = self.RNG.lognormal(mean=self.mean, sigma=self.stdev)

        # Raw Cholesky parameters (unconstrained)
        m11 = self.RNG.normal(0.0, 1.0)
        m22 = self.RNG.normal(0.0, 1.0)
        m33 = self.RNG.normal(0.0, 1.0)
        m21 = self.RNG.normal(0.0, 0.5)
        m31 = self.RNG.normal(0.0, 0.5)
        m32 = self.RNG.normal(0.0, 0.5)

        return {"v1": v1, "v2": v2, "v3": v3,
                "m11": m11, "m21": m21, "m31": m31,
                "m22": m22, "m32": m32, "m33": m33}

    def GBM_sim(self, v1, v2, v3,
                m11, m21, m31, m22, m32, m33,
                x0=np.array([100, 100, 100]),
                time=100 / 365, time_step=1 / 365):

        vols = np.array([v1, v2, v3])
        stdevs_D = np.diag(vols)

        # Cholesky factor for correlation (diagonal forced positive)
        M = np.array([
            [np.exp(m11), 0.0, 0.0],
            [m21, np.exp(m22), 0.0],
            [m31, m32, np.exp(m33)]
        ])

        # Build correlation matrix: normalize M M^T
        C = M @ M.T
        d = np.sqrt(np.diag(C))
        corr = C / (d[:, None] * d[None, :])

        # Covariance = D * corr * D
        sigma = stdevs_D @ corr @ stdevs_D

        b1, b2, b3 = 0.2, 0.4, -0.3

        x = x0
        motion = [x0]

        for _ in range(0, int(time / time_step) - 1):
            drift_coef = np.array([b1, b2, b3])
            correction = 0.5 * np.sum([sigma[:, j] ** 2 for j in range(0, 3)], axis=0)
            drift = drift_coef - correction
            timescaled_drift = drift * time_step
            random_shocks = sigma @ self.RNG.normal(scale=np.sqrt(time_step), size=3)
            dx = x * (timescaled_drift + random_shocks)
            x = x + dx
            motion.append(x)

        return dict(motion=np.asarray(motion))

    def __call__(self):
        drawn_prior = self.prior()
        motion = self.GBM_sim(**drawn_prior)

        return drawn_prior | motion


# ABM volatiilty-based simulation wrapper
class ABMVolWrapper:
    def __init__(self, seed : int = int(os.times()[4]), mean = 0.24621856131518247, stdev = 0.0049087692859631936*100):
        self.RNG = np.random.default_rng()
        self.mean = mean
        self.stdev = stdev

    def prior(self):
        # Generates a random draw from the prior

        v1 = self.RNG.lognormal(mean=self.mean, sigma=self.stdev)
        v2 = self.RNG.lognormal(mean=self.mean, sigma=self.stdev)
        v3 = self.RNG.lognormal(mean=self.mean, sigma=self.stdev)

        return {"v1": v1, "v2": v2, "v3": v3}

    def ABM_sim(self, v1, v2, v3, x0=np.array([100, 100, 100]), time=100 / 365, time_step=1 / 365):
        stdevs = np.array([v1, v2, v3])
        stdevs_D = np.diag(stdevs)
        correlation = np.array([[1.0, 0.4472136, 0.0],
                                [0.0, 1.0, 2.12132],
                                [0.0, 0.0, 1.0]])
        sigma = np.dot(stdevs_D, correlation, stdevs_D)

        b1, b2, b3 = 0.2, 0.4, -0.3  # drift terms

        x = x0.copy()
        motion = [x0]

        total_steps = int(time / time_step)

        for _ in range(total_steps - 1):
            drift = np.array([b1, b2, b3])
            timescaled_drift = drift * time_step

            random_shocks = sigma @ self.RNG.normal(scale=np.sqrt(time_step), size=3)

            dx = timescaled_drift + random_shocks  # ABM update
            x = x + dx
            motion.append(x)

        return dict(motion=np.asarray(motion))

    def __call__(self):
        drawn_prior = self.prior()
        motion = self.ABM_sim(**drawn_prior)

        return drawn_prior | motion


# ABM correlation-based simulation wrapper
class ABMCorrWrapper:
    def __init__(self, seed : int = int(os.times()[4]), corr_scale = 1):
        self.RNG = np.random.default_rng()
        self.scale = corr_scale

    def prior(self):
        rho12 = self.RNG.uniform(-self.scale, self.scale)
        rho13 = self.RNG.uniform(-self.scale, self.scale)
        rho23 = self.RNG.uniform(-self.scale, self.scale)

        return {"rho12": rho12, "rho13": rho13, "rho23": rho23}

    def ABM_sim(self, rho12, rho13, rho23, x0=np.array([100, 100, 100]), time=100 / 365, time_step=1 / 365):
        # Fixed volatilities to isolate correlation learning
        vols = np.array([0.25, 0.25, 0.25])
        D = np.diag(vols)

        # build correlation matrix from parameters
        corr = np.array([
            [1.0, rho12, rho13],
            [rho12, 1.0, rho23],
            [rho13, rho23, 1.0]
        ])

        # covariance
        sigma = D @ corr @ D

        b = np.array([0.2, 0.4, -0.3])
        x = x0.copy()
        motion = [x0]

        total_steps = int(time / time_step)

        for _ in range(total_steps - 1):
            drift_term = b * time_step
            shocks = sigma @ self.RNG.normal(scale=np.sqrt(time_step), size=3)

            dx = drift_term + shocks  # ABM: additive update
            x = x + dx
            motion.append(x)

        return dict(motion=np.asarray(motion))

    def __call__(self):
        drawn_prior = self.prior()
        motion = self.ABM_sim(**drawn_prior)

        return drawn_prior | motion


# ABM covariance matrix-based simulation wrapper
class ABMCovWrapper:
    def __init__(self, seed : int = int(os.times()[4]), mean = 0.24621856131518247, stdev = 0.0049087692859631936*100):
        self.RNG = np.random.default_rng()
        self.mean = mean
        self.stdev = stdev

    def prior(self):
        # Generates a random draw from the prior

        v1 = self.RNG.lognormal(mean=self.mean, sigma=self.stdev)
        v2 = self.RNG.lognormal(mean=self.mean, sigma=self.stdev)
        v3 = self.RNG.lognormal(mean=self.mean, sigma=self.stdev)

        # Raw Cholesky parameters (unconstrained)
        m11 = self.RNG.normal(0.0, 1.0)
        m22 = self.RNG.normal(0.0, 1.0)
        m33 = self.RNG.normal(0.0, 1.0)
        m21 = self.RNG.normal(0.0, 0.5)
        m31 = self.RNG.normal(0.0, 0.5)
        m32 = self.RNG.normal(0.0, 0.5)

        return {"v1": v1, "v2": v2, "v3": v3,
                "m11": m11, "m21": m21, "m31": m31,
                "m22": m22, "m32": m32, "m33": m33}

    def ABM_sim(self, v1, v2, v3,
                m11, m21, m31, m22, m32, m33,
                x0=np.array([100, 100, 100]),
                time=100 / 365, time_step=1 / 365):
        vols = np.array([v1, v2, v3])
        stdevs_D = np.diag(vols)

        # Cholesky factor for correlation (diagonal forced positive)
        M = np.array([
            [np.exp(m11), 0.0, 0.0],
            [m21, np.exp(m22), 0.0],
            [m31, m32, np.exp(m33)]
        ])

        # Build correlation matrix: normalize M M^T
        C = M @ M.T
        d = np.sqrt(np.diag(C))
        corr = C / (d[:, None] * d[None, :])

        # Covariance = D * corr * D
        sigma = stdevs_D @ corr @ stdevs_D

        b1, b2, b3 = 0.2, 0.4, -0.3

        x = x0.copy()
        motion = [x0]

        total_steps = int(time / time_step)

        for _ in range(total_steps - 1):
            drift = np.array([b1, b2, b3])
            timescaled_drift = drift * time_step

            random_shocks = sigma @ self.RNG.normal(scale=np.sqrt(time_step), size=3)

            dx = timescaled_drift + random_shocks  # ABM update (additive)
            x = x + dx
            motion.append(x)

        return dict(motion=np.asarray(motion))

    def __call__(self):
        drawn_prior = self.prior()
        motion = self.ABM_sim(**drawn_prior)

        return drawn_prior | motion

if __name__ == '__main__':
    myGVolWrap = GBMVolWrapper()
    myGCorrWrap = GBMCorrWrapper()
    myGCovWrap = GBMCovWrapper()

    print(myGVolWrap())
    print()
    print(myGCorrWrap())
    print()
    print(myGCovWrap())
    print()

    myAVolWrap = ABMVolWrapper()
    myACorrWrap = ABMCorrWrapper()
    myACovWrap = ABMCovWrapper()

    print(myAVolWrap())
    print()
    print(myACorrWrap())
    print()
    print(myACovWrap())