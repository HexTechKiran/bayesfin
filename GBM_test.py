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

RNG = np.random.default_rng(2025)
DRIFT_SCALE = 1

def prior():
    # Generates a random draw from the prior

    b1 = RNG.uniform(-DRIFT_SCALE, DRIFT_SCALE)
    b2 = RNG.uniform(-DRIFT_SCALE, DRIFT_SCALE)
    b3 = RNG.uniform(-DRIFT_SCALE, DRIFT_SCALE)

    return {"b1":b1, "b2":b2, "b3":b3}

def GBM_sim(b1, b2, b3, x0 = np.array([100, 100, 100]), time = 100/365, time_step = 1/365):
    sigma = np.array([[0.5, 0.1, 0.0],
                      [0.0, 0.1, 0.3],
                      [0.0, 0.0, 0.2]])

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
    motion = GBM_sim(prior_sample["b1"], prior_sample["b2"], prior_sample["b3"])["motion"]

    plt.plot(np.arange(0, time_, time_step_), [row[0] for row in motion])
    plt.plot(np.arange(0, time_, time_step_), [row[1] for row in motion])
    plt.plot(np.arange(0, time_, time_step_), [row[2] for row in motion])
    plt.show()
    """

    simulator = bf.simulators.make_simulator([prior, GBM_sim])

    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_time_series("motion")
        .concatenate(["b1", "b2", "b3"], into="inference_variables")
        .rename("motion", "summary_variables")
        .log(["inference_variables", "summary_variables"], p1=True)
    )

    summary_net = GRU()

    inference_net = bf.networks.CouplingFlow()

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net,
        standardize=None
    )

    train = workflow.simulate(8000)
    validation = workflow.simulate(500)

    history = workflow.fit_offline(data=train,
                                   epochs=100,
                                   batch_size=64,
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

    b1_truth = test_sims["b1"][0].item()
    b2_truth = test_sims["b2"][0].item()
    b3_truth = test_sims["b3"][0].item()
    truths = np.asarray([b1_truth, b2_truth, b3_truth])

    b1_samples = samples["b1"][0].flatten()
    b2_samples = samples["b2"][0].flatten()
    b3_samples = samples["b3"][0].flatten()
    out_samples = np.asarray([b1_samples, b2_samples, b3_samples]).T

    labels = ["b1", "b2", "b3"]

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

