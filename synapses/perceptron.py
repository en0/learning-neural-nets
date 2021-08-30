from .activator import deserialize_activator
from .typing import Vector, ActivatorInterface, PerceptronInterface


class Perceptron(PerceptronInterface):

    @classmethod
    def deserialize(cls, desc: dict) -> PerceptronInterface:
        return cls(weights=desc["w"], activator=deserialize_activator(desc["a"]))

    @property
    def weights(self) -> Vector:
        return self._weights[1:]

    @property
    def activator(self) -> ActivatorInterface:
        return self._activator

    @property
    def inputs(self) -> Vector:
        return self._last_inputs

    @property
    def output(self) -> float:
        return self._output

    def update_weights(self, weights: Vector) -> None:
        #print(f"Update Weights old=({self._weights}), new=({weights})")
        for i, weight in enumerate(weights):
            self._weights[i+1] = weight

    def serialize(self) -> dict:
        return {
            "w": self._weights,
            "a": self.activator.serialize(),
        }

    def signal(self, value: float) -> None:
        self._inputs.append(value)

    def activate(self, inputs: Vector = None) -> float:

        if inputs:
            self._inputs = inputs

        self._output = self._activator(sum(map(
            lambda v: v[0] * v[1],
            zip([1.0] + self._inputs, self._weights)
        )))

        for n in self._attached:
            n.signal(self.output)
        self._last_inputs = self._inputs.copy()
        self._inputs = []
        return self.output

    def attach_to(self, perceptron: PerceptronInterface):
        self._attached.append(perceptron)

    def __init__(
            self,
            weights: Vector,
            activator: ActivatorInterface,
    ):
        self._output: float = None
        self._inputs: Vector = []
        self._last_inputs: Vector = []

        self._weights = weights
        self._activator = activator
        self._attached = []
