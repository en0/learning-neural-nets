import unittest
from synapses.activator import construct_activator, ActivatorEnum, StepActivator


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


class ConstructActivatorTests(unittest.TestCase):
    def test_construct_step_activator(self):
        self.assertIsInstance(construct_activator(ActivatorEnum.Step), StepActivator)

    def test_construct_step_activator_sets_limit(self):
        activator = construct_activator(ActivatorEnum.Step, threshold=100)
        self.assertEqual(activator(100.1), 1)
        self.assertEqual(activator(99.9), 0)
