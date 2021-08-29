import unittest

from synapses.activator import ActivatorEnum
from synapses.model import TrainingModel


class BackpropagationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.example = {
            "name": "UnitTest",
            "desc": "From https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/",
            "layers": [
                [
                    {"b": 1.00, "w": [0.35, 0.15, 0.20], "a": {"e": ActivatorEnum.Logistic}},
                    {"b": 1.00, "w": [0.35, 0.25, 0.30], "a": {"e": ActivatorEnum.Logistic}}
                ],
                [
                    {"b": 1.00, "w": [0.60, 0.40, 0.45], "a": {"e": ActivatorEnum.Logistic}},
                    {"b": 1.00, "w": [0.60, 0.50, 0.55], "a": {"e": ActivatorEnum.Logistic}}
                ]
            ]
        }

    def test_simple_example(self):
        model = TrainingModel()
        model.deserialize(self.example)
        self.assertListEqual(model.predict([0.05, 0.1]), [0.7513650695523157, 0.7729284653214625])

    def test_total_error(self):
        model = TrainingModel()
        model.deserialize(self.example)
        print(model.fit([0.05, 0.10], [0.01, 0.99]))
        print(model.fit([0.05, 0.10], [0.01, 0.99]))
        #self.assertEqual(model._compute_partial_error(0.01, 0.7513650695523157), 0.7413650695523157)
        #self.assertEqual(model._compute_partial_activation_error(0.7513650695523157), 0.18681560180895948)

        #self.assertListEqual(model.predict([0.05, 0.1]), [0.7513650695523157, 0.7729284653214625])

