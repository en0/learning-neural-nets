import unittest
from synapses.activator import ActivatorEnum, StepActivator, deserialize_activator, LogisticActivator


class StepActivatorTests(unittest.TestCase):
    def test_int(self):
        a = StepActivator()
        self.assertEqual(a(1), 1)
        self.assertEqual(a(0), 0)

    def test_float(self):
        a = StepActivator()
        self.assertEqual(a(1.1), 1)
        self.assertEqual(a(-1.1), 0)

    def test_custom_threshold(self):
        a = StepActivator(10)
        self.assertEqual(a(1.1), 0)
        self.assertEqual(a(10.1), 1)

    def test_serialize(self):
        activator = StepActivator(100)
        self.assertDictEqual(activator.serialize(), {
            "e": ActivatorEnum.Step,
            "p": {"threshold": 100}
        })


class LogisticTest(unittest.TestCase):

    def test_serialize(self):
        self.assertDictEqual(LogisticActivator().serialize(), {"e": ActivatorEnum.Logistic})

    def test_deserialize(self):
        a = deserialize_activator({"e": ActivatorEnum.Logistic})
        self.assertIsInstance(a, LogisticActivator)

    def test_accuracy(self):
        a = LogisticActivator()
        self.assertEqual(0.5932699921071872, a(0.3775))


class ConstructActivatorTests(unittest.TestCase):

    def test_deserialize_step_activator(self):
        a = deserialize_activator({
            "e": ActivatorEnum.Step,
            "p": {"threshold": 100}
        })
        self.assertIsInstance(a, StepActivator)
        self.assertEqual(a(100.1), 1)
        self.assertEqual(a(99.9), 0)

