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
                    {"w": [1, -0.5, -0.5], "a": {"e": ActivatorEnum.Step, "p": {"threshold": 0}}},
                    # or
                    {"w": [0, 0.25, 0.25], "a": {"e": ActivatorEnum.Step, "p": {"threshold": 0}}}
                ],
                [
                    # and
                    {"w": [-0.25, 0.25, 0.25], "a": {"e": ActivatorEnum.Step, "p": {"threshold": 0}}}
                ]
            ]
        }

    def test_deserialize_sets_name(self):
        m = Model()
        m.deserialize(self.xor_model_description)
        self.assertEqual(m.name, self.xor_model_description["name"])

    def test_deserialize_sets_description(self):
        m = Model()
        m.deserialize(self.xor_model_description)
        self.assertEqual(m.desc, self.xor_model_description["desc"])

    def test_deserialize_creates_layers(self):
        m = Model()
        m.deserialize(self.xor_model_description)
        self.assertEqual(len(m._layers), 2)

    def test_deserialize_adds_correct_count_of_neurons(self):
        m = Model()
        m.deserialize(self.xor_model_description)
        self.assertEqual(len(m._layers[0]), 2)
        self.assertEqual(len(m._layers[1]), 1)

    def test_deserialize_adds_PerceptronInterface(self):
        m = Model()
        m.deserialize(self.xor_model_description)
        for layer in m._layers:
            for n in layer:
                self.assertIsInstance(n, PerceptronInterface)

    def test_deserialize_sets_perceptron_weights(self):
        m = Model()
        m.deserialize(self.xor_model_description)
        for layer, layer_desc in zip(m._layers, self.xor_model_description["layers"]):
            for n, desc in zip(layer, layer_desc):
                self.assertListEqual(n.weights, desc["w"])

    def test_deserialize_sets_perceptron_activator(self):
        m = Model()
        m.deserialize(self.xor_model_description)
        for layer in m._layers:
            for n in layer:
                self.assertIsInstance(n.activator, StepActivator)

    @patch("synapses.perceptron.Perceptron.attach_to")
    def test_deserialize_attaches_perceptrons(self, attach_method):
        m = Model()
        m.deserialize(self.xor_model_description)
        attach_method.assert_called()

    def test_prediction(self):
        m = Model()
        m.deserialize(self.xor_model_description)
        for i, ans in [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]:
            actual = m.predict(i)
            self.assertListEqual(actual, ans)

    def test_serialize_includes_name(self):
        m = Model()
        m.deserialize(self.xor_model_description)
        ans = m.serialize()
        self.assertEqual(ans["name"], self.xor_model_description["name"])

    def test_serialize_includes_desc(self):
        m = Model()
        m.deserialize(self.xor_model_description)
        ans = m.serialize()
        self.assertEqual(ans["desc"], self.xor_model_description["desc"])

    def test_serialize_includes_layers(self):
        m = Model()
        m.deserialize(self.xor_model_description)
        ans = m.serialize()
        self.assertEqual(len(ans["layers"]), len(self.xor_model_description["layers"]))

    def test_serialize_serialized_neurons(self):
        m = Model()
        m.deserialize(self.xor_model_description)
        ans = m.serialize()
        for layer, expected_layer in zip(ans["layers"], self.xor_model_description["layers"]):
            for neuron, expected_neuron in zip(layer, expected_layer):
                self.assertDictEqual(neuron, expected_neuron)
