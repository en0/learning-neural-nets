from typing import List
from .typing import Vector, Activator, Perceptron

class Neuron(Perceptron):
    def __init__(
        self,
        weights: Vector,
        activator: Activator,
        bias: float = 1
    ):
        self._output: float = None
        self._inputs: Vector = []

        self._weights = weights
        self._activator = activator
        self.bias = bias
        self._attached = []

    @property
    def output(self) -> float:
        return self._output

    def signal(self, input: float) -> None:
        self._inputs.append(input)

    def activate(self) -> None:
        if not self._inputs:
            raise Exception("No inputs")

        self._output = self._activator(sum(map(
            lambda v: v[0] * v[1],
            zip([self.bias] + self._inputs, self._weights)
        )))

        self._inputs = []

        for n in self._attached:
            n.signal(self.output)

    def attach_to(self, perceptron: Perceptron):
        self._attached.append(perceptron)
