import unittest
from unittest.mock import patch
from synapses.model import Model
from synapses.activator import ActivatorEnum, StepActivator
from synapses.perceptron import PerceptronInterface


# noinspection PyMethodMayBeStatic
class ModelTests(unittest.TestCase):

    def setUp(self) -> None:
        self.xor_model_description = {
            "name": "xor",
            "desc": "Compute Exclusive Or Operation on 2 inputs.",
            "layers": [
                [
                    # nand
                    {"b": 1, "w": [1, -0.5, -0.5], "a": {"e": ActivatorEnum.Step, "p": {"threshold": 0}}},
                    # or
                    {"b": 1, "w": [0, 0.25, 0.25], "a": {"e": ActivatorEnum.Step, "p": {"threshold": 0}}}
                ],
                [
                    # and
                    {"b": 1, "w": [-0.25, 0.25, 0.25], "a": {"e": ActivatorEnum.Step, "p": {"threshold": 0}}}
                ]
            ]
        }

    def test_load_sets_name(self):
        m = Model()
        m.load(self.xor_model_description)
        self.assertEqual(m.name, self.xor_model_description["name"])

    def test_load_sets_description(self):
        m = Model()
        m.load(self.xor_model_description)
        self.assertEqual(m.desc, self.xor_model_description["desc"])

    def test_load_creates_layers(self):
        m = Model()
        m.load(self.xor_model_description)
        self.assertEqual(len(m.layers), 2)

    def test_load_adds_correct_count_of_neurons(self):
        m = Model()
        m.load(self.xor_model_description)
        self.assertEqual(len(m.layers[0]), 2)
        self.assertEqual(len(m.layers[1]), 1)

    def test_load_adds_PerceptronInterface(self):
        m = Model()
        m.load(self.xor_model_description)
        for layer in m.layers:
            for n in layer:
                self.assertIsInstance(n, PerceptronInterface)

    def test_load_sets_perceptron_weights(self):
        m = Model()
        m.load(self.xor_model_description)
        for layer, layer_desc in zip(m.layers, self.xor_model_description["layers"]):
            for n, desc in zip(layer, layer_desc):
                self.assertListEqual(n.weights, desc["w"])

    def test_load_sets_perceptron_activator(self):
        m = Model()
        m.load(self.xor_model_description)
        for layer in m.layers:
            for n in layer:
                self.assertIsInstance(n.activator, StepActivator)

    def test_load_sets_perceptron_bias(self):
        m = Model()
        m.load(self.xor_model_description)
        for layer, layer_desc in zip(m.layers, self.xor_model_description["layers"]):
            for n, desc in zip(layer, layer_desc):
                self.assertEqual(n.bias, desc["b"])

    @patch("synapses.perceptron.Perceptron.attach_to")
    def test_load_attaches_perceptrons(self, attach_method):
        m = Model()
        m.load(self.xor_model_description)
        attach_method.assert_called()

    def test_prediction(self):
        m = Model()
        m.load(self.xor_model_description)
        for i, ans in [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]:
            actual = m.predict(i)
            self.assertListEqual(actual, ans)
