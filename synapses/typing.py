from abc import ABC, abstractmethod
from typing import List, Callable

Vector = List[float]
Activator = Callable[[float], float]

class Perceptron(ABC):
    @property
    @abstractmethod
    def output(self) -> float:
        ...

    @abstractmethod
    def activate(self, inputs: Vector) -> None:
        ...

    @abstractmethod
    def signal(self, input: float) -> None:
        ...

    @abstractmethod
    def attach_to(self, perceptron: "Perceptron"):
        ...
