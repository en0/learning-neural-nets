from unittest import TestCase
from synapses.neuron import Neuron


class TestNeuron(TestCase):

    def create_neuron(self, w, a):
        return Neuron(w, a)

    def test_simple(self):
        neuron = self.create_neuron([1,1,1], lambda x: x)
        neuron.signal(2)
        neuron.signal(3)
        print(neuron.activate())
