import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.factoryguard.exceptions import ModelingError

logger = logging.getLogger(__name__)

class FactoryGuardIsolationForest:
    def __init__(self, n_estimators: int = 100, contamination: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.train_scores_ = None
        self.fitted = False

    def fit(self, X: pd.DataFrame):
        """Fit Isolation Forest on training features."""
        logger.info("Fitting Isolation Forest model...")
        self.model.fit(X)
        
        # score_samples returns raw scores where lower (more negative) is more anomalous
        # We store these scores to calibrate the percentile mapping
        self.train_scores_ = self.model.score_samples(X)
        self.fitted = True
        logger.info("Isolation Forest fitted successfully.")
        return self

    def predict_anomaly_score_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Predict raw score (lower = more anomalous)."""
        if not self.fitted:
            raise ModelingError("Model must be fitted before prediction!")
        return self.model.score_samples(X)

    def predict_anomaly_score_normalized(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict normalized anomaly score from 0 to 100.
        0 = least abnormal (most normal), 100 = most abnormal (most anomalous).
        We calculate the percentile rank of the raw score against the training score distribution.
        """
        if not self.fitted:
            raise ModelingError("Model must be fitted before prediction!")
            
        raw_scores = self.predict_anomaly_score_raw(X)
        
        # Convert raw scores to normalized scores:
        # Since lower raw score is more anomalous, we want to map:
        # lowest raw scores -> 100 (high anomaly percentile)
        # highest raw scores -> 0 (low anomaly percentile)
        # We sort training scores ascending (most anomalous to least anomalous)
        sorted_train = np.sort(self.train_scores_)
        
        # searchsorted finds where raw_scores would be inserted to maintain order.
        # This tells us what proportion of training scores are MORE anomalous (lower) than the raw_score.
        # Percentile rank = (position in sorted_train / total_train) * 100
        # If raw_score is less than sorted_train[0], rank is 0 (which means 100% of train is less anomalous than it)
        # So: normalized_score = 100 - (percentile rank)
        ranks = np.searchsorted(sorted_train, raw_scores)
        percentile_ranks = (ranks / len(sorted_train)) * 100.0
        normalized_scores = 100.0 - percentile_ranks
        
        # Clip just in case
        return np.clip(normalized_scores, 0.0, 100.0)
