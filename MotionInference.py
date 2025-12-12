"""
This file is used to run simulation-based bayesian inference on the brownian motion simulators found in
Motion.py
"""

import os

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
from Motion import Motion

# GRU network to make our input usable for the inference network
class GRU(bf.networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gru = keras.layers.GRU(128, dropout=0.1)
        self.summary_stats = keras.layers.Dense(64)

    def call(self, time_series, **kwargs):
        summary = self.gru(time_series, training=kwargs.get("stage") == "training")
        summary = self.summary_stats(summary)
        return summary

if __name__ == "__main__":
    motion = Motion(simulator="geom", parameters="diag", with_prior=True)

    simulator = bf.simulators.make_simulator([lambda : motion()])

    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_time_series("motion")
        .concatenate(["b1", "b2", "b3", "v1", "v2", "v3", "m21", "m31", "m32"], into="inference_variables")
        .rename("motion", "summary_variables")
    )

    summary_net = GRU(dropout=0.1)

    # inference_net = bf.networks.CouplingFlow(transform="spline", depth=2, dropout=0.1)
    inference_net = bf.networks.FlowMatching(dropout=0.1)

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        summary_network=summary_net,
        inference_network=inference_net
    )

    train = workflow.simulate(8000)
    validation = workflow.simulate(300)

    history = workflow.fit_offline(data=train,
                                   epochs=100,
                                   batch_size=32,
                                   validation_data=validation)

    f = bf.diagnostics.plots.loss(history)

    # Save the workflow
    #workflow.approximator.save("gbm_drift_workflow/")

    plt.savefig("Motion_loss.png")

    num_datasets = 300
    num_samples = 1000

    # Simulate 300 scenarios
    print("Running simulations")
    test_sims = workflow.simulate(num_datasets)

    # Obtain num_samples posterior samples per scenario
    print("Sampling")
    samples = workflow.sample(conditions=test_sims, num_samples=num_samples)

    print("Making plots")
    f = bf.diagnostics.plots.recovery(samples, test_sims)

    plt.savefig("Motion_recoveries.png")

    labels = ["v1", "v2", "v3"]

    truths = np.asarray([test_sims[labels[0]][0].item(),
                         test_sims[labels[1]][0].item(),
                         test_sims[labels[2]][0].item()])

    out_samples = np.asarray([samples[labels[0]][0].flatten(),
                              samples[labels[1]][0].flatten(),
                              samples[labels[2]][0].flatten()]).T

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

    plt.savefig("Motion_hist.png")