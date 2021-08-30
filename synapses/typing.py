from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Callable, Tuple, Union

Vector = List[float]


class ActivatorEnum(Enum):
    Step = 1
    Pass = 2
    Logistic = 3


class ActivatorInterface(ABC):
    @abstractmethod
    def __call__(self, value: float) -> float:
        ...

    @abstractmethod
    def serialize(self) -> dict:
        ...

    @abstractmethod
    def compute_derivative(self, actual: float) -> float:
        ...


class PerceptronInterface(ABC):

    @property
    @abstractmethod
    def activator(self) -> ActivatorInterface:
        ...

    @property
    @abstractmethod
    def weights(self) -> Vector:
        ...

    @property
    @abstractmethod
    def inputs(self) -> Vector:
        ...

    @property
    @abstractmethod
    def output(self) -> float:
        ...

    @abstractmethod
    def activate(self, inputs: Vector = None) -> float:
        ...

    @abstractmethod
    def signal(self, value: float) -> None:
        ...

    @abstractmethod
    def attach_to(self, perceptron: "PerceptronInterface"):
        ...

    @abstractmethod
    def update_weights(self, weights: Vector) -> None:
        ...

    @abstractmethod
    def serialize(self) -> dict:
        ...

    @classmethod
    @abstractmethod
    def deserialize(cls, desc: dict) -> "PerceptronInterface":
        ...


class ModelInterface(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def desc(self) -> str:
        ...

    @abstractmethod
    def deserialize(self, model: dict) -> None:
        ...

    @abstractmethod
    def serialize(self) -> dict:
        ...

    @abstractmethod
    def predict(self, inputs: Vector) -> Vector:
        ...


class TrainingModelInterface(ModelInterface, ABC):

    @abstractmethod
    def fit(self, inputs: Vector, outputs: Vector) -> float:
        ...

    @abstractmethod
    def generate(
        self,
        name: str,
        desc: str,
        input_width: int,
        layer_width: List[int],
        layer_activators: Union[List[ActivatorInterface], ActivatorInterface],
        random_range: Tuple[float, float] = (0.5, 1.0)
    ) -> None:
        ...
