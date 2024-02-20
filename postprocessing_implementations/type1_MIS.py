# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:11:00 2023

@author: jwurtz
"""
from bloqade.utils import record
from bloqade.postprocess.problem.MIS_problem import unit_disk_maximum_independent_set
from bloqade.postprocess.solution.greedy_MIS import (
    greedy_MIS_solution,
    classical_greedy_MIS_solution,
)
from bloqade.postprocess.optimizer.base_optimizer import Optimizer
from bloqade.utils.visualize_UDG import get_UDG

import bloqade.postprocess as postprocess
import numpy as np
import os, tqdm, json

filename = os.path.join(os.getcwd(), "DATA", "type1_MIS_data.json")

# %%
record = postprocess.record.local_record(filename)
# record = recorder(local_file_path)

N = 3
seed_problem = 42
randgen = np.random.RandomState(seed_problem)
x, y = np.meshgrid(range(N), range(N))
pos = np.array([x.flatten(), y.flatten()]).T
perm = randgen.permutation(N**2)
positions = pos[perm[0 : int(N**2 * 0.7)]]


prototype_graph = os.path.join(os.getcwd(), "generate_figures_pos.json")
if seed_problem == 0:
    with open(prototype_graph, "r") as io:
        dat = json.load(io)
    positions = np.array(dat["positions"])


problem = unit_disk_maximum_independent_set(positions, 0.72)
solution = greedy_MIS_solution(
    problem, backend={"quantum": "python_emulate", "num_shots": 20}
)

record.put(problem)
record.put(solution)


quantum_parameters = {"time": 1.5, "initial_detuning": -15, "final_detuning": 15}
classical_parameters = {}
x_init = solution.flatten_parameters(quantum_parameters, classical_parameters)

optimize = Optimizer(
    solution,
    recorder=record,
    method="SPSA",
    initial_guess=x_init,
    max_iter=3,
    verbose=True,
)
# %%
optimize()
record.save()


# %% Asynchronously...
record2 = postprocess.record.local_record.load(filename)

problem2 = unit_disk_maximum_independent_set.from_file(record2)
solution2 = greedy_MIS_solution.from_file(problem2, record2)
optimize2 = Optimizer.from_file(solution2, record2)

# %%
# A moment of truth: I just re-used the (randomly selected) data from
# the clustering optimization instead of re-running. This means that there was
# a small amount of back-end code to glue the optimizer together that was
# not shown here.
# %%
solution2_c = classical_greedy_MIS_solution(
    problem2, backend={"quantum": "a potato", "num_shots": 100}
)

best_quantum_parameters, best_classical_parameters = solution2.unflatten_parameters(
    optimize2.tracker[np.argmin([q[1] for q in optimize2.tracker])][0]
)
dout = []
dout_c = []
for ctr in tqdm.trange(5):
    sol = solution2.get_solution(best_quantum_parameters, best_classical_parameters)
    sol_c = solution2_c.get_solution(best_quantum_parameters, best_classical_parameters)
    dout += list(sol)
    dout_c += list(sol_c)

vals = []
vals_c = []
for sol in dout:
    vals.append(problem2.cost(sol))
for sol in dout_c:
    vals_c.append(problem2.cost(sol))

# %%

import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = ["serif"]
plt.close("all")

RED = "#C2477F"
PURPLE = "#6437FF"
YELLOW = "#edff1a"
YELLOW2 = "#a8b512"
GREEN = "#55de79"
BLACK = "#878787"


minval = int(np.min(vals + vals_c))
maxval = int(np.max(vals + vals_c))
P_q = np.histogram(
    vals, bins=maxval - minval + 2, range=[minval - 1, maxval + 1], density=True
)
P_c = np.histogram(
    vals_c, bins=maxval - minval + 2, range=[minval - 1, maxval + 1], density=True
)


plt.figure(figsize=(8, 6))
plt.bar(P_q[1][0:-1] - 0.2, P_q[0], width=0.4, color=RED, label="quantum")
plt.bar(P_c[1][0:-1] + 0.2, P_c[0], width=0.4, color=PURPLE, label="classical")
plt.legend()
plt.xlabel("mIS size")
plt.ylabel("Probability")
plt.legend()
plt.axis([37.6, 47.4, 0, 0.6])
