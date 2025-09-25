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

RNG = np.random.default_rng(2025)

def prior():
    # Generates a random draw from the joint prior

    b1 = RNG.uniform(-1, 1)
    b2 = RNG.uniform(-1, 1)
    b3 = RNG.uniform(-1, 1)

    return {"b1":b1, "b2":b2, "b3":b3}

def GBM_sim(b1, b2, b3, x0, time: int, time_step: float):
    sigma = np.array([[0.5, 0.1, 0.0],
                      [0.0, 0.1, 0.3],
                      [0.0, 0.0, 0.2]])

    x = x0

    motion = [x0]

    for i in range(0, int(time/time_step) - 1):
        drift_coef = np.array([b1, b2, b3])
        correction = 0.5*np.sum([sigma[:, j]**2 for j in range(0, 3)], axis=0)
        drift = drift_coef - correction
        timescaled_drift = drift * time_step
        shock_matrix = sigma * RNG.normal(scale=1*time_step, size=(1, 3))
        random_shocks = np.sum([shock_matrix[:, 0], shock_matrix[:, 1], shock_matrix[:, 2]], axis=0)
        dx = x * (timescaled_drift + random_shocks)
        x = x + dx
        motion.append(x)

    return motion

if __name__ == "__main__":
    prior_sample = prior()
    motion = GBM_sim(prior_sample["b1"], prior_sample["b2"], prior_sample["b3"], np.array([100, 100, 100]), 100/365, 1/365)

    plt.plot(np.arange(0, 100/365, 1/365), [row[0] for row in motion])
    plt.plot(np.arange(0, 100/365, 1/365), [row[1] for row in motion])
    plt.plot(np.arange(0, 100/365, 1/365), [row[2] for row in motion])
    plt.show()