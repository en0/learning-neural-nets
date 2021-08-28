from typing import List

from synapses.activator import construct_activator, ActivatorEnum
from synapses.perceptron import Perceptron
from synapses.typing import TrainingModelInterface, ModelInterface, Vector, PerceptronInterface


class Model(ModelInterface):

    def __init__(self):
        self.name: str = None
        self.desc: str = None
        self.layers: List[List[PerceptronInterface]] = []

    def load(self, model: dict) -> None:
        self.name = model["name"]
        self.desc = model["desc"]
        for layer_desc in model["layers"]:
            layer = []
            for neuron_desc in layer_desc:
                activator = construct_activator(
                    enum=neuron_desc["a"]["e"],
                    **neuron_desc["a"].get("p", {})
                )
                neuron = Perceptron(
                    weights=neuron_desc["w"].copy(),
                    activator=activator,
                    bias=neuron_desc["b"]
                )
                layer.append(neuron)

            # Attach the previous layer to the current layer
            for n1 in (self.layers[-1] if self.layers else []):
                for n in layer:
                    n1.attach_to(n)

            self.layers.append(layer)

    def predict(self, inputs: Vector) -> Vector:
        for neuron in self.layers[0]:
            neuron.activate(inputs)
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.activate()
        return [n.output for n in self.layers[-1]]

class TrainingModel(Model, TrainingModelInterface):
    ...

