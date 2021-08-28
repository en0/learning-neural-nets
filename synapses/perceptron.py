from .typing import Vector, Activator, PerceptronInterface


class Perceptron(PerceptronInterface):
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
        self._bias = bias
        self._attached = []

    @property
    def weights(self) -> Vector:
        return self._weights

    @property
    def activator(self) -> Activator:
        return self._activator

    @property
    def bias(self) -> Activator:
        return self._bias

    @property
    def output(self) -> float:
        return self._output

    def signal(self, input: float) -> None:
        self._inputs.append(input)

    def activate(self, inputs: Vector = None) -> float:

        if inputs:
            self._inputs = inputs

        if not self._inputs:
            raise Exception("No inputs")

        self._output = self._activator(sum(map(
            lambda v: v[0] * v[1],
            zip([self._bias] + self._inputs, self._weights)
        )))

        self._inputs = []

        for n in self._attached:
            n.signal(self.output)

        return self.output

    def attach_to(self, perceptron: PerceptronInterface):
        self._attached.append(perceptron)
