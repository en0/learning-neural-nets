from typing import List, Union, Tuple
from random import uniform

from synapses.perceptron import Perceptron
from synapses.typing import TrainingModelInterface, ModelInterface, Vector, PerceptronInterface, ActivatorInterface


def _serialize(name, desc, layers) -> dict:
    return {
        "name": name,
        "desc": desc,
        "layers": [
            [n.serialize() for n in layer]
            for layer in layers
        ],
    }

class Model(ModelInterface):

    @property
    def name(self) -> str:
        return self._name

    @property
    def desc(self) -> str:
        return self._desc

    def deserialize(self, model: dict) -> None:
        self._reset()
        self._name = model["name"]
        self._desc = model["desc"]
        for layer_desc in model["layers"]:
            layer = [Perceptron.deserialize(d) for d in layer_desc]

            # Attach the previous layer to the current layer
            for n1 in (self._layers[-1] if self._layers else []):
                for n in layer:
                    n1.attach_to(n)

            self._layers.append(layer)

    def serialize(self) -> dict:
        return _serialize(self._name, self._desc, self._layers)

    def predict(self, inputs: Vector) -> Vector:
        for neuron in self._layers[0]:
            neuron.activate(inputs)
        for layer in self._layers[1:]:
            for neuron in layer:
                neuron.activate()
        return [n.output for n in self._layers[-1]]

    def _reset(self):
        self._name: str = None
        self._desc: str = None
        self._layers: List[List[PerceptronInterface]] = []

    def __init__(self):
        self._name: str = None
        self._desc: str = None
        self._layers: List[List[PerceptronInterface]] = []


class TrainingModel(Model, TrainingModelInterface):

    def fit(self, inputs: Vector, ideals: Vector) -> float:

        # Map of new weights for each neurons
        update_map = {}

        # Intermediate values that will propagate backwards through the network
        intermediate_values = [[] for _ in range(len(self._layers))]

        actual = self.predict(inputs)
        total_error = self._compute_total_error(actual, ideals)

        # compute new weights and intermediate errors from outputs
        for neuron, ideal in zip(self._layers[-1], ideals):
            update_weights, intermediate_parts = [], []
            o_error = self._compute_partial_error(ideal, neuron.output)
            a_error = neuron.activator.compute_derivative(neuron.output)
            for o_input, o_weight in zip(neuron.inputs, neuron.weights):
                delta = o_error * a_error * o_input
                o_update = o_weight - self._learning_rate * delta
                update_weights.append(o_update)
                intermediate_parts.append(o_error * a_error * o_weight)
            intermediate_values[-1].append(intermediate_parts)
            update_map[neuron] = update_weights

        # compute new weights and intermediate values each hidden layer
        # staring with the one closest to the output layer.
        for layer_index in range(len(self._layers) - 2, -1, -1):
            for neuron in self._layers[layer_index]:
                # Obtain the intermediate values from the layer after the current layer
                iv = intermediate_values[layer_index + 1]
                update_weights, intermediate_parts = [], []
                a_error = neuron.activator.compute_derivative(neuron.output)
                for o_input, o_weight, *ip in zip(neuron.inputs, neuron.weights, *iv):
                    o_error = sum(ip)
                    delta = o_error * a_error * o_input
                    o_update = o_weight - self._learning_rate * delta
                    intermediate_parts.append(o_error * a_error * o_weight)
                    update_weights.append(o_update)
                intermediate_values[layer_index].append(intermediate_parts)
                update_map[neuron] = update_weights

        # Update Weights
        for neuron, weights in update_map.items():
            neuron.update_weights(weights)

        return total_error

    def generate(
            self,
            name: str,
            desc: str,
            input_width: int,
            layer_width: List[int],
            layer_activators: Union[List[ActivatorInterface], ActivatorInterface],
            random_range: Tuple[float, float] = (0.5, 1.0)
    ) -> None:

        def as_iterator(thing):
            while not isinstance(thing, list):
                yield thing
            for o in thing:
                yield o

        layer_desc = zip(
            [input_width] + layer_width[:-1],
            layer_width,
            as_iterator(layer_activators)
        )

        _layers = []
        for i_width, l_width, activator in layer_desc:
            layer = [
                Perceptron(
                    weights=[0.0] + [uniform(*random_range) for i in range(i_width)],
                    activator=activator,
                ) for _ in range(l_width)
            ]
            _layers.append(layer)
        self.deserialize(_serialize(name, desc, _layers))

    def _compute_total_error(self, actuals: Vector, ideals: Vector) -> float:
        total_error = 0
        for actual, ideal in zip(actuals, ideals):
            output_error = 0.5 * (ideal - actual) ** 2
            total_error += output_error
        return total_error

    def _compute_partial_error(self, ideal: float, actual: float) -> float:
        partial_error = -(ideal - actual)
        return partial_error

    def __init__(self, learning_rate: float = 0.5):
        super().__init__()
        self._learning_rate = learning_rate
