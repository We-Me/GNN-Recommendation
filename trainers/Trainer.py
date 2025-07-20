import abc
from typing import Any


class BaseTrainer(object):
    def __init__(self, option: dict[str, Any]):
        self.option = option

    @abc.abstractmethod
    def train(self, *kargs, **kwargs):
        pass

    @abc.abstractmethod
    def eval(self, *kargs, **kwargs):
        pass

    @abc.abstractmethod
    def save_model(self, *kargs, **kwargs):
        pass
