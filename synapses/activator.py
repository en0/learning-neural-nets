## https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
import math

from synapses.typing import ActivatorInterface, ActivatorEnum


class StepActivator(ActivatorInterface):

    def serialize(self) -> dict:
        return {
            "e": ActivatorEnum.Step,
            "p": {"threshold": self._threshold}
        }

    def compute_derivative(self, actual: float) -> float:
        return 0

    def __init__(self, threshold: float = 0):
        self._threshold = threshold

    def __call__(self, value: float) -> float:
        return 1 if value > self._threshold else 0


class PassActivator(ActivatorInterface):

    def serialize(self) -> dict:
        return {"e": ActivatorEnum.Pass}

    def compute_derivative(self, actual: float) -> float:
        raise NotImplementedError()

    def __call__(self, value: float) -> float:
        return value


class LogisticActivator(ActivatorInterface):

    def serialize(self) -> dict:
        return {"e": ActivatorEnum.Logistic}

    def compute_derivative(self, actual: float) -> float:
        return actual * (1 - actual)

    def __call__(self, value: float) -> float:
        return 1.0 / (1 + math.e ** -value)


def deserialize_activator(desc: dict) -> ActivatorInterface:
    return {
        ActivatorEnum.Step: StepActivator,
        ActivatorEnum.Pass: PassActivator,
        ActivatorEnum.Logistic: LogisticActivator,
    }[desc["e"]](**desc.get("p", {}))
