from unittest import TestCase
from synapses.perceptron import Perceptron


class TestPerceptron(TestCase):

    def create_neuron(self, w, a):
        return Perceptron(w, a)

    def signal_and_activate(self, n, inputs):
        for i in inputs:
            n.signal(i)
        n.activate()
        return n.output

    def test_simple(self):
        neuron = self.create_neuron([1, 1, 1], lambda x: x)
        ans = self.signal_and_activate(neuron, [2, 3])
        self.assertEqual(int(ans), 6)

    def test_attached(self):
        n1 = self.create_neuron([1, 1, 1], lambda x: x)
        n2 = self.create_neuron([1, 1, 1], lambda x: x)
        n3 = self.create_neuron([1, 1, 1], lambda x: x)

        n2.attach_to(n1)
        n3.attach_to(n1)

        # n2 should return 6
        n2.signal(2)
        n2.signal(3)
        n2.activate()

        # n3 should return 6
        n3.signal(2)
        n3.signal(3)
        n3.activate()

        # n1 gets 6, 6 + bias. Should return 13
        n1.activate()

        self.assertEqual(n1.output, 13)

    def test_and(self):
        n = self.create_neuron([-0.25, 0.25, 0.25], lambda x: 1 if x > 0 else 0)
        ans = self.signal_and_activate(n, [1, 1])
        self.assertEqual(ans, 1)

        ans = self.signal_and_activate(n, [0, 0])
        self.assertEqual(ans, 0)

        ans = self.signal_and_activate(n, [0, 1])
        self.assertEqual(ans, 0)

        ans = self.signal_and_activate(n, [1, 0])
        self.assertEqual(ans, 0)

    def test_or(self):
        n = self.create_neuron([0, 0.25, 0.25], lambda x: 1 if x > 0 else 0)
        ans = self.signal_and_activate(n, [1, 1])
        self.assertEqual(ans, 1)

        ans = self.signal_and_activate(n, [0, 1])
        self.assertEqual(ans, 1)

        ans = self.signal_and_activate(n, [1, 0])
        self.assertEqual(ans, 1)

        ans = self.signal_and_activate(n, [0, 0])
        self.assertEqual(ans, 0)

    def test_nand(self):
        n = self.create_neuron([1, -0.5, -0.5], lambda x: 1 if x > 0 else 0)
        ans = self.signal_and_activate(n, [1, 1])
        self.assertEqual(ans, 0)

        ans = self.signal_and_activate(n, [0, 0])
        self.assertEqual(ans, 1)

        ans = self.signal_and_activate(n, [0, 1])
        self.assertEqual(ans, 1)

        ans = self.signal_and_activate(n, [1, 0])
        self.assertEqual(ans, 1)

    def test_xor(self):
        n_nand = self.create_neuron([1, -0.5, -0.5], lambda x: 1 if x > 0 else 0)
        n_and = self.create_neuron([-0.25, 0.25, 0.25], lambda x: 1 if x > 0 else 0)
        n_or = self.create_neuron([0, 0.25, 0.25], lambda x: 1 if x > 0 else 0)

        n_nand.attach_to(n_and)
        n_or.attach_to(n_and)

        for a, b, ans in [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]:
            n_nand.signal(a)
            n_nand.signal(b)

            n_or.signal(a)
            n_or.signal(b)

            n_nand.activate()
            n_or.activate()
            n_and.activate()

            self.assertEqual(n_and.output, ans)
