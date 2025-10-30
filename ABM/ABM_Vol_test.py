"""
Simulates a multivariable geometric brownian motion with 3 variables, and performs inference on
the volatility vector used as an input to create the volatility coefficient matrix.
Outputs metrics on the posterior distributions generated.
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
import corner
import jax

RNG = np.random.default_rng(int(os.times()[4]))
VOL_MEAN = 0.24621856131518247
VOL_STDEV = 0.0049087692859631936

def prior():
    # Generates a random draw from the prior

    v1 = RNG.lognormal(mean=VOL_MEAN, sigma=VOL_STDEV)
    v2 = RNG.lognormal(mean=VOL_MEAN, sigma=VOL_STDEV)
    v3 = RNG.lognormal(mean=VOL_MEAN, sigma=VOL_STDEV)

    return {"v1":v1, "v2":v2, "v3":v3}

def ABM_sim(v1, v2, v3, x0 = np.array([100, 100, 100]), time = 100/365, time_step = 1/365):
    stdevs = np.array([v1, v2, v3])
    stdevs_D = np.diag(stdevs)
    correlation = np.array([[1.0,     0.4472136, 0.0],
                            [0.0,     1.0,       2.12132],
                            [0.0,     0.0,       1.0]])
    sigma = np.dot(stdevs_D, correlation, stdevs_D)

    b1, b2, b3 = 0.2, 0.4, -0.3   # drift terms

    x = x0.copy()
    motion = [x0]

    total_steps = int(time / time_step)

    for _ in range(total_steps - 1):
        drift = np.array([b1, b2, b3])
        timescaled_drift = drift * time_step

        random_shocks = sigma @ RNG.normal(scale=np.sqrt(time_step), size=3)

        dx = timescaled_drift + random_shocks   # ABM update
        x = x + dx
        motion.append(x)

    return dict(motion=np.asarray(motion))

# GRU network to make our input usable for the inference network
class GRU(bf.networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gru = keras.layers.GRU(64, dropout=0.1)
        self.summary_stats = keras.layers.Dense(16)

    def call(self, time_series, **kwargs):
        summary = self.gru(time_series, training=kwargs.get("stage") == "training")
        summary = self.summary_stats(summary)
        return summary

if __name__ == "__main__":
    """
    prior_sample = prior()
    time_ = 100/365
    time_step_ = 1/365
    motion = ABM_sim(prior_sample["v1"], prior_sample["v2"], prior_sample["v3"])["motion"]

    plt.plot(np.arange(0, time_, time_step_), [row[0] for row in motion])
    plt.plot(np.arange(0, time_, time_step_), [row[1] for row in motion])
    plt.plot(np.arange(0, time_, time_step_), [row[2] for row in motion])
    plt.show()
    """

    simulator = bf.simulators.make_simulator([prior, ABM_sim])

    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_time_series("motion")
        .concatenate(["v1", "v2", "v3"], into="inference_variables")
        .rename("motion", "summary_variables")
        .log(["inference_variables", "summary_variables"], p1=True)
    )

    summary_net = bf.networks.TimeSeriesNetwork(dropout=0.1)

    inference_net = bf.networks.CouplingFlow(transform="spline", depth=2, dropout=0.1)

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
        standardize=None
    )

    train = workflow.simulate(8000)
    validation = workflow.simulate(300)

    history = workflow.fit_offline(data=train,
                                   epochs=100,
                                   batch_size=32,
                                   validation_data=validation)

    f = bf.diagnostics.plots.loss(history)

    plt.show()

    num_datasets = 300
    num_samples = 1000

    # Simulate 300 scenarios
    test_sims = workflow.simulate(num_datasets)

    # Obtain num_samples posterior samples per scenario
    samples = workflow.sample(conditions=test_sims, num_samples=num_samples)

    f = bf.diagnostics.plots.recovery(samples, test_sims)

    b1_truth = test_sims["v1"][0].item()
    b2_truth = test_sims["v2"][0].item()
    b3_truth = test_sims["v3"][0].item()
    truths = np.asarray([b1_truth, b2_truth, b3_truth])

    b1_samples = samples["v1"][0].flatten()
    b2_samples = samples["v2"][0].flatten()
    b3_samples = samples["v3"][0].flatten()
    out_samples = np.asarray([b1_samples, b2_samples, b3_samples]).T

    labels = ["v1", "v2", "v3"]

    d = out_samples.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(8, 8))

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.set_facecolor("white")  # set background blue
                ax.hist(out_samples[:, i], bins=40, histtype="step", color="lightblue")
                ax.axvline(truths[i], color="red")
                ax.set_xlabel(labels[i])
            elif i < j:
                ax.set_facecolor("midnightblue")  # set background blue
                h = ax.hist2d(out_samples[:, j], out_samples[:, i],
                              bins=50, cmap="viridis")
                ax.plot(truths[j], truths[i], "o", color="red")
            else:
                ax.axis("off")

    plt.tight_layout()
    plt.show()

    plt.show()

