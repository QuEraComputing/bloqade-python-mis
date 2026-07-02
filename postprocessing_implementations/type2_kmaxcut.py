# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:12:26 2023

@author: jwurtz
"""


from bloqade.postprocess.problem.k_maxcut import k_maxcut
from bloqade.postprocess.solution.spectral_kmaxcut import spectral_kmaxcut_solution
from bloqade.postprocess.optimizer.base_optimizer import Optimizer
from bloqade.utils.visualize_UDG import get_UDG
from bloqade.utils.greedy_flip import greedy_flip

import bloqade.postprocess as postprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tqdm

filename = os.path.join(os.getcwd(), "DATA", "type2_kmaxcut_data.json")

# %%
record = postprocess.record.local_record(filename)


N = 14
seed_problem = 0
randgen = np.random.RandomState(seed_problem)
# x, y = np.meshgrid(range(N), range(N))
# pos = np.array([x.flatten(), y.flatten()]).T
# perm = randgen.permutation(N**2)
# positions = pos[perm[0:int(N**2 * .7)]]


prototype_graph = os.path.join(os.getcwd(), "generate_figures_pos.json")
if seed_problem == 0:
    with open(prototype_graph, "r") as io:
        dat = json.load(io)
    positions = np.array(dat["positions"])


graph = get_UDG(positions, 0.72)

problem = k_maxcut(graph, positions, 3, which="MAX")
solution = spectral_kmaxcut_solution(
    problem, backend={"quantum": "braket_aquila", "num_shots": 100}
)

record.put(problem)
record.put(solution)

# classical_parameters = {"w0":-np.sin(0.9)*np.cos(0.4618),
# "w1":-np.cos(0.9)*np.cos(0.4618),"w2":-np.sin(0.4618)}
# Optimized for a particular instance
classical_parameters = {
    "w0": -0.7778392547192745,
    "w1": -0.6474488067050066,
    "w2": -0.8643199521322924,
}

quantum_parameters = {
    "time1": 0.05 + 0.05 * randgen.rand(),
    "time2": 0.15 + 0.05 * randgen.rand(),
    "time3": 0.2 + 0.1 * randgen.rand(),
    "delta1": 0.0 + 3 * randgen.rand(),
    "delta2": 0.0 + 3 * randgen.rand(),
    "delta3": 0.0 + 3 * randgen.rand(),
}

x_init = solution.flatten_parameters(quantum_parameters, classical_parameters)

optimize = Optimizer(
    solution,
    recorder=record,
    method="Bayesian",
    initial_guess=x_init,
    max_iter=100,
    fixed_indices=[0, 1, 2, 3, 4, 5],
    bounds=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
    verbose=True,
)

optimize_q = Optimizer(
    optimize,
    recorder=None,
    method="Random",
    initial_guess=[x_init[ind] for ind in [0, 1, 2, 3, 4, 5]],
    max_iter=20,
    bounds=[
        (0.05, 0.15),
        (0.05, 0.2),
        (0.05, 0.2),
        (0.0, 10.0),
        (0.0, 10.0),
        (0.0, 10.0),
    ],
    verbose=False,
)
# %%

# raise BaseException("STOP")
optimize_q()
record.put(problem)
record.put(solution)
record.save()


# %% Asynchronously...
record2 = postprocess.record.local_record.load(filename)

problem2 = k_maxcut.from_file(record2)
solution2 = spectral_kmaxcut_solution.from_file(problem2, record2)
optimize2 = Optimizer.from_file(solution2, record2)

# %% Analyze results by generating Q, C, plus greedy-post-processed
best_quantum_parameters, best_classical_parameters = solution2.unflatten_parameters(
    optimize2.tracker[np.argmin([q[1] for q in optimize2.tracker])][0]
)
dout = []
solns = []
for ctr in tqdm.trange(512):
    sol = solution2.get_solution(best_quantum_parameters, best_classical_parameters)
    sol_c = np.random.choice(problem2.nclusters, size=problem2.graph.number_of_nodes())

    sol_QG = greedy_flip(problem2, sol)
    sol_CG = greedy_flip(problem2, sol_c)

    dout.append(
        [
            problem2.cost(sol),
            problem2.cost(sol_c),
            problem2.cost(sol_QG),
            problem2.cost(sol_CG),
        ]
    )
    solns.append([sol, sol_QG])
dout = np.array(dout)
# %% Plot

import matplotlib.pyplot as plt  # noqa: F811, E402

plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = ["serif"]
plt.close("all")

RED = "#C2477F"
PURPLE = "#6437FF"
YELLOW = "#edff1a"
YELLOW2 = "#a8b512"
GREEN = "#55de79"
BLACK = "#878787"


fig = plt.figure(figsize=(12, 6))

a, b = np.histogram(dout[:, 0], bins=np.max(dout), range=[0, np.max(dout)])
b = problem2.graph.number_of_edges() - b
plt.plot(b[0:-1], a / sum(a), ":", color=RED, linewidth=3, label="Quantum")
plt.fill_between(b[0:-1], a / sum(a), color=RED, alpha=0.5)

a, b = np.histogram(dout[:, 1], bins=np.max(dout), range=[0, np.max(dout)])
b = problem2.graph.number_of_edges() - b
plt.plot(b[0:-1], a / sum(a), ":", color=PURPLE, linewidth=3, label="Classical")
plt.fill_between(b[0:-1], a / sum(a), color=PURPLE, alpha=0.5)

a, b = np.histogram(dout[:, 2], bins=np.max(dout), range=[0, np.max(dout)])
b = problem2.graph.number_of_edges() - b
plt.plot(b[0:-1], a / sum(a), color=RED, linewidth=3, label="Quantum + Greedy")
plt.fill_between(b[0:-1], a / sum(a), color=RED, alpha=0.5)

a, b = np.histogram(dout[:, 3], bins=np.max(dout), range=[0, np.max(dout)])
b = problem2.graph.number_of_edges() - b
plt.plot(b[0:-1], a / sum(a), color=PURPLE, linewidth=3, label="Classical + Greedy")
plt.fill_between(b[0:-1], a / sum(a), color=PURPLE, alpha=0.5)


plt.axis(
    [
        problem2.graph.number_of_edges() - np.max(dout),
        problem2.graph.number_of_edges(),
        0.0,
        plt.axis()[3],
    ]
)
plt.legend()
plt.xlabel("Edge color violations")
plt.ylabel("Probability density")

# %%
# Also generate the clustering figure
features, centers, partition, correlation = solution2._get_features_centers_partition(
    best_quantum_parameters, best_classical_parameters
)

# Weighted correlation function
plt.figure(figsize=(8, 6))
ax = plt.subplot(1, 1, 1)
plt.imshow(correlation, cmap="plasma", vmin=-1.5, vmax=1.5)
ax.set_xticks(np.arange(0, 125, 20))

# Features
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")
ax.scatter(
    features[partition == 0, 0],
    features[partition == 0, 1],
    features[partition == 0, 2],
    s=50,
)
ax.scatter(
    features[partition == 1, 0],
    features[partition == 1, 1],
    features[partition == 1, 2],
    s=50,
)
ax.scatter(
    features[partition == 2, 0],
    features[partition == 2, 1],
    features[partition == 2, 2],
    s=50,
)
ax.axis("off")
# Rotate by hand to get a good projection...
