import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from src.factoryguard.exceptions import ModelingError

logger = logging.getLogger(__name__)

class ProbabilityCalibrator:
    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.calibrator = None
        self.fitted = False

    def fit(self, y_probs: np.ndarray, y_true: np.ndarray):
        """Fit calibration model mapping uncalibrated probabilities (y_probs) to true labels (y_true)."""
        logger.info(f"Fitting probability calibrator using method: {self.method}...")
        
        # Reshape for sklearn
        X = y_probs.reshape(-1, 1)
        
        if self.method == "sigmoid":
            # Platt scaling: Fit logistic regression on the probabilities
            self.calibrator = LogisticRegression(C=1.0, random_state=42)
            self.calibrator.fit(X, y_true)
        elif self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(y_probs, y_true)
        else:
            raise ModelingError(f"Unknown calibration method: {self.method}")
            
        self.fitted = True
        logger.info("Calibrator fitted successfully.")
        return self

    def calibrate(self, y_probs: np.ndarray) -> np.ndarray:
        """Calibrate probabilities."""
        if not self.fitted:
            raise ModelingError("Calibrator must be fitted before calling calibrate!")
            
        if self.method == "sigmoid":
            X = y_probs.reshape(-1, 1)
            # Return class 1 probability
            return self.calibrator.predict_proba(X)[:, 1]
        elif self.method == "isotonic":
            return self.calibrator.predict(y_probs)
        return y_probs
