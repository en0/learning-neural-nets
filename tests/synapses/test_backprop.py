import unittest
from random import uniform

from synapses.activator import ActivatorEnum, LogisticActivator
from synapses.model import TrainingModel


class BackpropagationTest(unittest.TestCase):

    def setUp(self) -> None:
        self.example = {
            "name": "UnitTest",
            "desc": "From https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/",
            "layers": [
                [
                    {"w": [0.35, 0.15, 0.20], "a": {"e": ActivatorEnum.Logistic}},
                    {"w": [0.35, 0.25, 0.30], "a": {"e": ActivatorEnum.Logistic}}
                ],
                [
                    {"w": [0.60, 0.40, 0.45], "a": {"e": ActivatorEnum.Logistic}},
                    {"w": [0.60, 0.50, 0.55], "a": {"e": ActivatorEnum.Logistic}}
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
        self.assertEqual(model.fit([0.05, 0.10], [0.01, 0.99]), 0.2983711087600027)
        self.assertEqual(model.fit([0.05, 0.10], [0.01, 0.99]), 0.29102774132153775)

    def test_generation(self):
        model = TrainingModel()
        model.generate(
            name="xor",
            desc="Compute XOR",
            input_width=2,
            layer_width=[2, 1],
            layer_activators=LogisticActivator(),
        )
        self.assertEqual(len(model._layers), 2)
        self.assertEqual(len(model._layers[0]), 2)
        self.assertEqual(len(model._layers[1]), 1)
        for layer in model._layers:
            for n in layer:
                self.assertIsInstance(n.activator, LogisticActivator)

    def test_generate_and_train(self):
        model = TrainingModel(learning_rate=1)
        logistic = LogisticActivator()
        model.generate(
            name="xor",
            desc="Compute XOR",
            input_width=2,
            layer_width=[2, 1],
            layer_activators=logistic,
            random_range=(0.02, 0.05)
        )

        ins = [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ]

        e = []
        for inputs, ideal in ins:
            e.append(model.fit(inputs, ideal))

        while sum(e) / len(e) > 0.0855:
            e = []
            for inputs, ideal in ins:
                e.append(model.fit(inputs, ideal))
            print(sum(e) / len(e))

        self.assertGreaterEqual(model.predict([1, 0])[0], 0.5, "Failed 1")
        self.assertGreaterEqual(model.predict([0, 1])[0], 0.5, "Failed 2")
        self.assertLess(model.predict([1, 1])[0], 0.5, "Failed 3")
        self.assertLess(model.predict([0, 0])[0], 0.5, "Failed 4")
