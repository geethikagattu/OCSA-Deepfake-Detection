from abc import ABC, abstractmethod


class AdaptationMethod(ABC):
    """
    Base interface for all adaptation strategies.

    Every adaptation method should implement:
        adapt(model, data_loader)
    """

    name = "BASE"

    @abstractmethod
    def adapt(self, model, data_loader):
        """
        Adapt the detector using incoming data.

        Returns
        -------
        model:
            Adapted detector.
        """
        raise NotImplementedError


class WaitAdaptation(AdaptationMethod):
    """
    WAIT means do not modify the detector.
    """

    name = "WAIT"

    def adapt(self, model, data_loader=None):
        return model
