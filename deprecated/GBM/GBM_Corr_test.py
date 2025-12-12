"""
Simulates a multivariable geometric brownian motion with 3 variables, and performs inference on
a vector of correlations used as an input to create the volatility coefficient matrix.
Outputs metrics on the posterior distributions generated.
"""


import os

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bayesflow as bf
import keras
import jax

RNG = np.random.default_rng(int(os.times()[4]))
CORR_SCALE = 1

# Priors for correlation parameters, drawn in unconstrained space
def prior():
    rho12 = RNG.uniform(0, CORR_SCALE)
    rho13 = RNG.uniform(0, CORR_SCALE)
    rho23 = RNG.uniform(0, CORR_SCALE)

    return {"rho12": rho12, "rho13": rho13, "rho23": rho23}

def GBM_sim(rho12, rho13, rho23, x0=np.array([100,100,100]), time=100/365, time_step=1/365):
    # Fixed volatilities to isolate correlation learning
    vols = np.array([0.25, 0.25, 0.25])
    D = np.diag(vols)

    # build correlation matrix from parameters
    corr = np.array([
        [1.0,   rho12, rho13],
        [rho12, 1.0,   rho23],
        [rho13, rho23, 1.0]
    ])

    # covariance
    sigma = D @ corr @ D

    b = np.array([0.2, 0.4, -0.3])
    x = x0.copy()
    motion = [x0]

    for _ in range(int(time/time_step)-1):
        correction = 0.5 * np.sum([sigma[:, j]**2 for j in range(3)], axis=0)
        drift = b - correction
        drift_term = drift * time_step
        shocks = sigma @ RNG.normal(scale=np.sqrt(time_step), size=3)
        dx = x * (drift_term + shocks)
        x = x + dx
        motion.append(x)

    return dict(motion=np.asarray(motion))

# GRU summary network
class GRU(bf.networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gru = keras.layers.GRU(64, dropout=0.1)
        self.dense = keras.layers.Dense(16)

    def call(self, x, **kwargs):
        s = self.gru(x, training=kwargs.get("stage")=="training")
        return self.dense(s)

if __name__ == "__main__":

    simulator = bf.simulators.make_simulator([prior, GBM_sim])

    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_time_series("motion")
        .concatenate(["rho12","rho13","rho23"], into="inference_variables")
        .rename("motion", "summary_variables")
        .log(["summary_variables"], p1=True)
    )

    summary_net = bf.networks.TimeSeriesTransformer(dropout=0.1)
    inference_net = bf.networks.CouplingFlow(transform="spline", depth=2, dropout=0.1)

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
        standardize=None
    )

    train = workflow.simulate(8000)
    val = workflow.simulate(300)

    history = workflow.fit_offline(
        data=train, epochs=100, batch_size=32, validation_data=val
    )

    bf.diagnostics.plots.loss(history)
    plt.savefig("GBM_Corr_loss.png")

    num_datasets = 300
    num_samples = 1000
    tests = workflow.simulate(num_datasets)
    samples = workflow.sample(conditions=tests, num_samples=num_samples)

    f = bf.diagnostics.plots.recovery(samples, tests)
    plt.savefig("GBM_Corr_recovery.png")

    # corner-style plot
    r12_true = tests["rho12"][0].item()
    r13_true = tests["rho13"][0].item()
    r23_true = tests["rho23"][0].item()
    truths = np.array([r12_true, r13_true, r23_true])

    r12_s = samples["rho12"][0].flatten()
    r13_s = samples["rho13"][0].flatten()
    r23_s = samples["rho23"][0].flatten()
    out = np.array([r12_s, r13_s, r23_s]).T
    labels = ["rho12","rho13","rho23"]

    d = out.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(8,8))

    for i in range(d):
        for j in range(d):
            ax = axes[i,j]
            if i == j:
                ax.hist(out[:,i], bins=40, histtype="step")
                ax.axvline(truths[i], color="red")
                ax.set_xlabel(labels[i])
            elif i < j:
                ax.hist2d(out[:,j], out[:,i], bins=50, cmap="viridis")
                ax.plot(truths[j], truths[i], "o", color="red")
            else:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig("GBM_Corr_corner.png")
