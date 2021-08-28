from abc import ABC, abstractmethod
from typing import List, Callable

Vector = List[float]
Activator = Callable[[float], float]


class PerceptronInterface(ABC):

    @property
    @abstractmethod
    def activator(self) -> Activator:
        ...

    @property
    @abstractmethod
    def bias(self) -> Activator:
        ...

    @property
    @abstractmethod
    def weights(self) -> Vector:
        ...

    @property
    @abstractmethod
    def output(self) -> float:
        ...

    @abstractmethod
    def activate(self, inputs: Vector = None) -> float:
        ...

    @abstractmethod
    def signal(self, input: float) -> None:
        ...

    @abstractmethod
    def attach_to(self, perceptron: "PerceptronInterface"):
        ...


class ModelInterface(ABC):
    @abstractmethod
    def load(self, model: dict) -> None:
        ...

    @abstractmethod
    def predict(self, inputs: Vector) -> Vector:
        ...


class TrainingModelInterface(ModelInterface, ABC):
    ...
