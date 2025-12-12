"""
Simulates a multivariable geometric brownian motion with 3 variables, and performs inference on
a cholesky decomposed volatility coefficient matrix.
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
import os
import jax

RNG = np.random.default_rng(int(os.times()[4]))

VOL_MEAN = 0.24621856131518247
VOL_STDEV = 0.0049087692859631936 * 100

# Prior over volatilities and Cholesky correlation params
def prior():
    v1 = RNG.lognormal(mean=VOL_MEAN, sigma=VOL_STDEV)
    v2 = RNG.lognormal(mean=VOL_MEAN, sigma=VOL_STDEV)
    v3 = RNG.lognormal(mean=VOL_MEAN, sigma=VOL_STDEV)

    # Raw Cholesky parameters (unconstrained)
    m11 = RNG.normal(0.0, 1.0)
    m22 = RNG.normal(0.0, 1.0)
    m33 = RNG.normal(0.0, 1.0)
    m21 = RNG.normal(0.0, 0.5)
    m31 = RNG.normal(0.0, 0.5)
    m32 = RNG.normal(0.0, 0.5)

    return {"v1":v1, "v2":v2, "v3":v3,
            "m11":m11, "m21":m21, "m31":m31,
            "m22":m22, "m32":m32, "m33":m33}

def GBM_sim(v1, v2, v3,
            m11, m21, m31, m22, m32, m33,
            x0 = np.array([100, 100, 100]),
            time = 100/365, time_step = 1/365):

    vols = np.array([v1, v2, v3])
    stdevs_D = np.diag(vols)

    # Cholesky factor for correlation (diagonal forced positive)
    M = np.array([
        [np.exp(m11),     0.0,         0.0],
        [m21,             np.exp(m22), 0.0],
        [m31,             m32,         np.exp(m33)]
    ])

    # Build correlation matrix: normalize M M^T
    C = M @ M.T
    d = np.sqrt(np.diag(C))
    corr = C / (d[:,None] * d[None,:])

    # Covariance = D * corr * D
    sigma = stdevs_D @ corr @ stdevs_D

    b1, b2, b3 = 0.2, 0.4, -0.3

    x = x0
    motion = [x0]

    for _ in range(0, int(time/time_step) - 1):
        drift_coef = np.array([b1, b2, b3])
        correction = 0.5*np.sum([sigma[:, j]**2 for j in range(0, 3)], axis=0)
        drift = drift_coef - correction
        timescaled_drift = drift * time_step
        random_shocks = sigma @ RNG.normal(scale=np.sqrt(time_step), size=3)
        dx = x * (timescaled_drift + random_shocks)
        x = x + dx
        motion.append(x)

    return dict(motion=np.asarray(motion))

# GRU summary network
class GRU(bf.networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gru = keras.layers.GRU(64, dropout=0.1)
        self.summary_stats = keras.layers.Dense(16)

    def call(self, time_series, **kwargs):
        s = self.gru(time_series, training=kwargs.get("stage") == "training")
        s = self.summary_stats(s)
        return s


if __name__ == "__main__":

    simulator = bf.simulators.make_simulator([prior, GBM_sim])

    variables = ["v1","v2","v3",
                 "m11","m21","m31",
                 "m22","m32","m33"]

    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_time_series("motion")
        .concatenate(variables, into="inference_variables")
        .rename("motion", "summary_variables")
        #.log(["inference_variables", "summary_variables"], p1=True)
    )

    #summary_net = bf.networks.TimeSeriesNetwork(dropout=0.1)
    summary_net = GRU(dropout=0.1)

    inference_net = bf.networks.CouplingFlow(transform="spline", depth=2, dropout=0.1)

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net
        #standardize=None
    )

    train = workflow.simulate(8000)
    validation = workflow.simulate(300)

    history = workflow.fit_offline(
        data=train,
        epochs=100,
        batch_size=32,
        validation_data=validation
    )

    f = bf.diagnostics.plots.loss(history)
    plt.savefig("GBM_Cov_loss.png")

    num_datasets = 300
    num_samples = 1000

    test_sims = workflow.simulate(num_datasets)
    samples = workflow.sample(conditions=test_sims, num_samples=num_samples)

    f = bf.diagnostics.plots.recovery(samples, test_sims)
    plt.savefig("GBM_Cov_recoveries.png")

    # Example marginal vol posteriors for first simulation
    truths = np.array([test_sims["v1"][0].item(),
                       test_sims["v2"][0].item(),
                       test_sims["v3"][0].item()])

    v1_s = samples["v1"][0].flatten()
    v2_s = samples["v2"][0].flatten()
    v3_s = samples["v3"][0].flatten()
    out = np.asarray([v1_s, v2_s, v3_s]).T
    labels = ["v1", "v2", "v3"]

    d = out.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(8, 8))

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(out[:, i], bins=40, histtype="step")
                ax.axvline(truths[i], color="red")
                ax.set_xlabel(labels[i])
            elif i < j:
                h = ax.hist2d(out[:, j], out[:, i], bins=40)
                ax.plot(truths[j], truths[i], "o", color="red")
            else:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig("GBM_Cov_hist.png")
