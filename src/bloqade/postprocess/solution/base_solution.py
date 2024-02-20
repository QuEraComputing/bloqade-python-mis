# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:10:45 2023

@author: jwurtz
"""
from bloqade.postprocess.problem.base_problem import Problem
import bloqade.postprocess as postprocess
from abc import ABC, abstractmethod, abstractstaticmethod, abstractproperty

import bloqade
import json


class Solution(ABC):
    """
    The `Solution` class serves as an abstract base class for defining various solution strategies
    or ansatze for a given optimization problem. The class provides a framework for generating
    parameterized quantum circuits, computing candidate solutions, and evaluating objective functions.

    Attributes:
    - problem (Problem): The specific problem instance for which the solution is being computed.
    - _job_tape (dict): A dictionary to store the results of quantum jobs.
    - _backend (dict): Information about the quantum backend being used.
    """

    def __init__(self, problem: Problem, backend: dict):
        """
        Initializes the Solution class with a given problem instance and backend information.
        """

        self.problem = problem
        self._job_tape = {}
        self._backend = backend

    def __call__(self, x):
        """
        Compute the value of the objective function given parameters x.
        x are a flattened set of parameters ordered by [quantum_keys] + [classical_keys]
        """

        quantum, classical = self.unflatten_parameters(x)

        cost = self.objective(quantum, classical)

        return cost

    @abstractmethod
    def ansatz(self, quantum_parameters: dict, classical_parameters: dict = {}):
        """
        Generates parameterized quantum circuits based on provided quantum and classical parameters.

        Args:
        - quantum_parameters (dict): Parameters for the quantum circuit.
        - classical_parameters (dict, optional): Classical parameters for the circuit.

        Returns:
        - Quantum circuit or the appropriate representation.
        """
        return None

    @abstractmethod
    def get_solution(self, quantum_parameters: dict, classical_parameters: dict = {}):
        """
        Computes candidate solutions to the problem using quantum calls.

        Args:
        - quantum_parameters (dict): Quantum parameters.
        - classical_parameters (dict, optional): Classical parameters.

        Returns:
        - Candidate solution or solutions for the problem.
        """
        return None

    @abstractmethod
    def objective(
        self, quantum_parameters: dict, classical_parameters: dict = {}
    ) -> float:
        """
        Computes the objective function value based on quantum and classical parameters.

        Args:
        - quantum_parameters (dict): Quantum parameters.
        - classical_parameters (dict, optional): Classical parameters.

        Returns:
        - float: Computed objective function value.
        """
        return None

    @abstractproperty
    def classical_parameters(self) -> list[str]:
        """
        List of keys for classical_parameters
        """
        return []

    @abstractproperty
    def quantum_parameters(self) -> list[str]:
        """
        Provides the list of keys for classical parameters.

        Returns:
        - list[str]: List of keys representing classical parameters.
        """
        return []

    def flatten_parameters(self, quantum, classical) -> list:
        return [quantum[key] for key in self.quantum_parameters] + [
            classical[key] for key in self.classical_parameters
        ]

    def unflatten_parameters(self, x):
        # Forge the flattened array into the correct keyed variables
        quantum_keys = self.quantum_parameters
        classical_keys = self.classical_parameters

        # print(quantum_keys)
        # print(classical_keys)

        if len(x) != len(quantum_keys) + len(classical_keys):
            raise ValueError("Input parameters are the wrong length")
        quantum = {key: x[i] for i, key in enumerate(quantum_keys)}
        classical = {
            key: x[i + len(quantum_keys)] for i, key in enumerate(classical_keys)
        }
        return quantum, classical

    def submit(self, quantum_parameters: dict):
        """
        Submits a quantum job based on the provided quantum parameters.

        Args:
        - quantum_parameters (dict): Quantum parameters.

        Returns:
        - Quantum job result or appropriate representation.
        """
        key = tuple(quantum_parameters.items())

        # Check if the job has been done before
        if "_job_tape" in self.__dict__:
            if key in self._job_tape:
                return self._job_tape[key]

        # program.braket(...) -> program.braket.aquila().submit(...)
        # program.braket_simulator(...) -> program.braket.local_emulator(...).submit(...)
        # program.quera(...) -> program.quera.aquila(...).submit()

        # Else, submit
        if self._backend["quantum"] == "braket_aquila":
            submitted_job = (
                self.ansatz(quantum_parameters)
                .braket.aquila()
                .run_async(self._backend["num_shots"])
            )
            print("Job submitted via BraKet to Aquila!")
            print(submitted_job)
        elif self._backend["quantum"] == "braket_emulate":
            submitted_job = (
                self.ansatz(quantum_parameters)
                .braket.local_emulator()
                .run(self._backend["num_shots"])
            )
        elif self._backend["quantum"] == "python_emulate":
            submitted_job = (
                self.ansatz(quantum_parameters)
                .bloqade.python()
                .run(self._backend["num_shots"])
            )
        elif self._backend["quantum"] == "internal_aquila":
            # fname = "C:\\Users\\qpc1\\Documents\\GitHub\\hybrid_jobs\\hybrid_algorithms\\prod_config.json"
            fname = "C:\\Users\\jwurtz\\Downloads\\prod_config.json"
            submitted_job = (
                self.ansatz(quantum_parameters)
                .quera.device(fname)
                .run_async(self._backend["num_shots"])
            )
        else:
            raise BaseException("Bad backend")

        # Save to tape
        key = tuple(quantum_parameters.items())
        if "_job_tape" not in self.__dict__:
            self._job_tape = {key: submitted_job}
        else:
            self._job_tape[key] = submitted_job

        # self.to_file("tape.json")
        return submitted_job

    def get_bitstrings(self, quantum_parameters: dict):
        """
        Collects the bitstrings/results from the quantum job.

        Args:
        - quantum_parameters (dict): Quantum parameters.

        Returns:
        - Bitstrings or equivalent quantum results.
        """
        job = self.submit(quantum_parameters)

        bstr = job.report().bitstrings(False)

        if isinstance(job, bloqade.task.batch.LocalBatch):
            pass
        else:
            isready = len(job.get_completed_tasks().tasks) == len(job.tasks)
            for k, bb in enumerate(bstr):
                if len(bb) != job.tasks[k].nshots:
                    isready = False
            if isready == True:
                return job.report().bitstrings(False)

            print("Waiting for job to be complete...")
            job.pull()
            print("Remote results pulled!")

        return job.report().bitstrings(False)

    def __str__(self) -> None:
        """
        Serializes and saves the job tape to a specified file.

        Args:
        - recorder : a postprocess.record class which saves data locally or remotely
        """
        payload = {}
        payload["job_tape"] = [
            (key, json.loads(job.json())) for key, job in self._job_tape.items()
        ]
        payload["backend"] = self._backend
        return json.dumps(payload)

    def to_file(self, recorder: postprocess.record):
        recorder.put(self)

    @classmethod
    def from_file(cls, problem, recorder: postprocess.record.record) -> None:
        """
        Loads the job tape from a specified record and updates the current job tape.

        Args:
        - recorder : a postprocess.record class which saves data locally or remotely
        """

        payload = json.loads(recorder.get(cls))
        print("Loading from file!")

        out = cls(problem, backend=payload["backend"])
        out._job_tape = {}
        for key, val in payload["job_tape"]:
            # key = {a:tuple(b) for a,b in dict(key).items()}.items()
            key = {a: b for a, b in dict(key).items()}.items()

            job = bloqade.loads(json.dumps(val))
            out._job_tape[tuple(key)] = job

        return out
