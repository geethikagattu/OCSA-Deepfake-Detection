from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd


@dataclass
class AdaptationRecord:
    """One historical adaptation event."""

    experiment_id: str
    domain_source: str
    domain_target: str
    adaptation_method: str

    # Detector state before adaptation.
    uncertainty: Optional[float] = None
    feature_distance: Optional[float] = None
    drift_score: Optional[float] = None

    # Performance.
    metric_before: Optional[float] = None
    metric_after: Optional[float] = None

    # Calculated outcome.
    delta_metric: Optional[float] = None
    outcome: Optional[str] = None


class AdaptationHistory:
    def __init__(self):
        self.records = []

    def add(self, record: AdaptationRecord):
        self.records.append(asdict(record))

    def to_dataframe(self):
        return pd.DataFrame(self.records)

    def save(self, path):
        self.to_dataframe().to_csv(path, index=False)
