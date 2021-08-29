from .activator import deserialize_activator
from .typing import Vector, ActivatorInterface, PerceptronInterface


class Perceptron(PerceptronInterface):

    @classmethod
    def deserialize(cls, desc: dict) -> PerceptronInterface:
        return cls(weights=desc["w"],
                   activator=deserialize_activator(desc["a"]),
                   bias=desc["b"])

    @property
    def weights(self) -> Vector:
        # Remove bias from output
        return self._weights[1:]

    @property
    def activator(self) -> ActivatorInterface:
        return self._activator

    @property
    def bias(self) -> ActivatorInterface:
        return self._bias

    @property
    def inputs(self) -> Vector:
        return self._inputs

    @property
    def output(self) -> float:
        return self._output

    def update_weights(self, weights: Vector) -> None:
        for i, weight in enumerate(weights):
            self._weights[i+1] = weight

    def serialize(self) -> dict:
        return {
            "w": self._weights,
            "b": self._bias,
            "a": self.activator.serialize(),
        }

    def signal(self, value: float) -> None:
        self._inputs.append(value)

    def activate(self, inputs: Vector = None) -> float:

        if inputs:
            self._inputs = inputs

        #if not self._inputs:
            #raise Exception("Perceptron received no inputs.")

        self._output = self._activator(sum(map(
            lambda v: v[0] * v[1],
            zip([self._bias] + self._inputs, self._weights)
        )))

        #self._inputs = []

        for n in self._attached:
            n.signal(self.output)

        return self.output

    def attach_to(self, perceptron: PerceptronInterface):
        self._attached.append(perceptron)

    def __init__(
            self,
            weights: Vector,
            activator: ActivatorInterface,
            bias: float = 1
    ):
        self._output: float = None
        self._inputs: Vector = []

        self._weights = weights
        self._activator = activator
        self._bias = bias
        self._attached = []
