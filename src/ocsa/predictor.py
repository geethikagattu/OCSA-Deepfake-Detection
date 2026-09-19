import numpy as np


class OCSAPredictor:
    """
    Initial OCSA decision layer.

    This is intentionally simple. Later this will learn:

        detector state + incoming data + candidate action
            -> expected outcome
    """

    ACTIONS = [
        "WAIT",
        "T2A",
        "FINETUNE",
        "FREDA",
    ]

    OUTCOMES = [
        "BENEFICIAL",
        "NEUTRAL",
        "HARMFUL",
    ]

    def __init__(self):
        self.model = None

    def fit(self, X, y):
        """
        Train the OCSA outcome predictor.

        X:
            State/action features.

        y:
            Observed outcome.
        """
        self.model = "TODO"

    def predict_outcome(self, X):
        """
        Predict whether an adaptation action is beneficial, neutral, or harmful.
        """
        if self.model is None:
            raise RuntimeError("OCSA predictor has not been trained.")

        raise NotImplementedError

    def select_action(self, state):
        """
        Select the adaptation action with the highest expected benefit.
        """
        raise NotImplementedError
