from enum import Enum

from synapses.typing import Activator


class StepActivator(Activator):

    def __init__(self, threshold: float = 0):
        self._threshold = threshold

    def __call__(self, value: float) -> float:
        return 1 if value > self._threshold else 0


class ActivatorEnum(Enum):
    Step = 1


def construct_activator(enum: ActivatorEnum, **kwargs):
    return {
        ActivatorEnum.Step: StepActivator
    }[enum](**kwargs)
