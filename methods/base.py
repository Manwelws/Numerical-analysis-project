from abc import ABC, abstractmethod


class RootFinding(ABC):
    def __init__(self, f, tol=1e-6, max_iter=100):
        self.f = f
        self.tol = tol
        self.max_iter = max_iter
        self.iterations = []

    @abstractmethod
    def solve(self, **kwargs):
        pass

    def _record(self, **row):
        self.iterations.append(row)  # list of dictionaries


class LinearSystem(ABC):
    def __init__(self):
        self.steps = []

    @abstractmethod
    def solve(self, A: list, B: list) -> dict:
        pass

    def _record(self, description: str, matrix_state):
        self.steps.append({"description": description, "matrix_state": matrix_state})
