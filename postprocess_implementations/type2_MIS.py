# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:12:18 2023

@author: jwurtz
"""


from bloqade.postprocess.problem.MIS_problem import unit_disk_maximum_independent_set
from bloqade.postprocess.solution.tempering_MIS import tempering_MIS_solution,classical_tempering_MIS_solution
from bloqade.postprocess.optimizer.base_optimizer import Optimizer


import bloqade.postprocess as postprocess
import numpy as np
import os
import json
import tqdm

filename = os.path.join(os.getcwd(),"DATA","23_12_1_t2_MIS.json")

#%%
record = postprocess.record.local_record(filename)
#record = recorder(local_file_path)

# Generate a random King's subgraph instance
N = 14
seed_problem = 0
randgen = np.random.RandomState(seed_problem)
x,y = np.meshgrid(range(N), range(N))
pos = np.array([x.flatten(), y.flatten()]).T
perm = randgen.permutation(N**2)
positions = pos[perm[0:int(N**2 * .7)]] # 30% random dropout
        
if seed_problem == 0:
    with open("C:\\Users\\jwurtz\\Documents\\PYTHON\\postprocess_paper\\generate_figures_pos.json","r") as io:
        dat = json.load(io)
    positions = np.array(dat["positions"])


problem = unit_disk_maximum_independent_set(positions,0.72)
solution = tempering_MIS_solution(problem, backend = {"quantum": "braket_aquila", "num_shots": 100})

record.put(problem)
record.put(solution)


quantum_parameters = {"time": 3.5, "initial_detuning": -15, "final_detuning": 50}
classical_parameters = {"epochs":3,"beta":1000.0}
x_init = solution.flatten_parameters(quantum_parameters,classical_parameters)


optimize = Optimizer(solution, recorder = record,
                     method = "Random",
                     initial_guess = x_init,
                     max_iter =40,
                     verbose = True,
                     fixed_indices = [3,4],
                     bounds = [(2.5,3.9),
                               (-15.,-8.0),
                               (25.,60.)])
#%%
#raise BaseException("STOP")
optimize([3,1000])
record.put(optimize)
record.put(solution)
record.save()


#%% Asynchronously...
record2 = postprocess.record.local_record.load(filename)

problem2 = unit_disk_maximum_independent_set.from_file(record2)
solution2 = tempering_MIS_solution.from_file(problem2,record2)
optimize2 = Optimizer.from_file(solution2,record2)

#%% Plot results...
classical_vals = []
quantum_vals = []

#best_quantum_parameters, best_classical_parameters = quantum_parameters,classical_parameters
best_quantum_parameters, best_classical_parameters = solution2.unflatten_parameters(optimize2.tracker[np.argmin([q[1] for q in optimize2.tracker])][0])


solution_c = classical_tempering_MIS_solution(problem2,{})
dout = []
dout_c=[]
for ctr in tqdm.trange(256):
    sol_C = solution_c.get_solution({"sampling":"all"},best_classical_parameters)
    sol_Q = solution2.get_solution(best_quantum_parameters,best_classical_parameters)
    classical_vals.append([problem2.cost(ss) for ss in sol_C])
    quantum_vals.append([problem2.cost(ss) for ss in sol_Q])
    

classical_vals = np.array(classical_vals)
quantum_vals = np.array(quantum_vals)

ave_c = np.average(classical_vals,0)
std_c = np.std(classical_vals,0)
ave_q = np.average(quantum_vals,0)
std_q = np.std(quantum_vals,0)

xx = np.linspace(0,best_classical_parameters["epochs"],len(ave_c))

#%%
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = ["serif"]
plt.close('all')

RED = "#C2477F"
PURPLE = "#6437FF"
YELLOW = "#edff1a"
YELLOW2= "#a8b512"
GREEN  = "#55de79"
BLACK  = "#878787"


plt.figure(figsize=(8,6))


plt.fill_between(xx,ave_c-std_c,ave_c+std_c,color=PURPLE,alpha=0.5)
plt.fill_between(xx,ave_q-std_q,ave_q+std_q,color=RED,alpha=0.5)
plt.plot(xx,ave_c,"--",linewidth=3,color=PURPLE,label="Classical")
plt.plot(xx,ave_q,"--",linewidth=3,color=RED,label="Quantum")




plt.xlabel("Epoch")
plt.ylabel("Average mIS size")
plt.legend()
plt.axis([0,max(xx),35,plt.axis()[3]])


vals = [qv[-1] for qv in quantum_vals]
vals_c = [cv[-1] for cv in classical_vals]

minval = int(np.min(vals + vals_c))
maxval = int(np.max(vals+vals_c))
P_q = np.histogram(vals,bins=maxval-minval+2,range=[minval-1,maxval+1],density=True)
P_c = np.histogram(vals_c,bins=maxval-minval+2,range=[minval-1,maxval+1],density=True)


plt.figure(figsize=(8,6))
plt.bar(P_q[1][0:-1]-0.2,P_q[0],width=0.4,color=RED,label="quantum")
plt.bar(P_c[1][0:-1]+0.2,P_c[0],width=0.4,color=PURPLE,label="classical")
plt.legend()
plt.xlabel("mIS size")
plt.ylabel("Probability")
plt.legend()
plt.axis([43.6,47.4,0,0.8])

    