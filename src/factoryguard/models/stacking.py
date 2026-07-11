import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.factoryguard.exceptions import ModelingError

logger = logging.getLogger(__name__)

class ChronologicalStackingClassifier:
    def __init__(self, base_models: dict, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model or make_pipeline(StandardScaler(), LogisticRegression(C=1.0, random_state=42))
        self.fitted_base_models = {}
        self.fitted = False

    def train_oof_and_fit_meta(self, df_train: pd.DataFrame, feature_cols: list[str], target_col: str, n_folds: int = 3):
        """
        Train base models on temporal folds, gather out-of-fold validation predictions,
        and fit the meta-model on them.
        """
        logger.info(f"Starting chronological out-of-fold stacking with {n_folds} folds...")
        
        # Sort chronologically to make sure fold boundaries are chronological
        df_train = df_train.sort_values(by='datetime').reset_index(drop=True)
        n_samples = len(df_train)
        
        # Determine temporal splits
        # Fold 1: Train 0 to 40%, Val 40% to 60%
        # Fold 2: Train 0 to 60%, Val 60% to 80%
        # Fold 3: Train 0 to 80%, Val 80% to 100%
        split_pcts = [0.4, 0.6, 0.8]
        
        oof_predictions = {name: np.zeros(n_samples) for name in self.base_models.keys()}
        oof_mask = np.zeros(n_samples, dtype=bool)
        
        for fold, split_pct in enumerate(split_pcts):
            train_end_idx = int(n_samples * split_pct)
            val_end_idx = int(n_samples * (split_pct + 0.2)) if fold < len(split_pcts) - 1 else n_samples
            
            logger.info(f"Fold {fold+1}: training on rows 0-{train_end_idx}, validating on rows {train_end_idx}-{val_end_idx}...")
            
            # Extract split data
            X_tr_fold = df_train.loc[0:train_end_idx-1, feature_cols]
            y_tr_fold = df_train.loc[0:train_end_idx-1, target_col]
            X_val_fold = df_train.loc[train_end_idx:val_end_idx-1, feature_cols]
            
            # Train each base model and predict
            for name, model_class in self.base_models.items():
                # Instantiate fresh clone/estimator
                from sklearn.base import clone
                try:
                    # LGBM or XGB clone might need specific handling, cloned standard scikit-learn is fine
                    # For non-sklearn estimators, we can just instantiate from self.base_models or copy
                    import copy
                    est = copy.deepcopy(model_class)
                except Exception:
                    est = clone(model_class)
                    
                est.fit(X_tr_fold, y_tr_fold)
                
                # Predict probabilities
                if hasattr(est, "predict_proba"):
                    probs = est.predict_proba(X_val_fold)[:, 1]
                else:
                    # Fallback if model doesn't have predict_proba
                    probs = est.predict(X_val_fold)
                    
                oof_predictions[name][train_end_idx:val_end_idx] = probs
                
            oof_mask[train_end_idx:val_end_idx] = True
            
        # Fit meta-model on OOF predictions of base models
        # Only fit on the parts of the dataset where we generated OOF predictions
        X_meta = pd.DataFrame({name: oof_predictions[name][oof_mask] for name in self.base_models.keys()})
        y_meta = df_train.loc[oof_mask, target_col]
        
        logger.info("Fitting meta-model on out-of-fold predictions...")
        self.meta_model.fit(X_meta, y_meta)
        
        # 6. Retrain base models on the entire training set
        logger.info("Retraining all base models on the entire training period...")
        X_all_train = df_train[feature_cols]
        y_all_train = df_train[target_col]
        
        for name, model in self.base_models.items():
            logger.info(f"Retraining {name} on full train set...")
            import copy
            est = copy.deepcopy(model)
            est.fit(X_all_train, y_all_train)
            self.fitted_base_models[name] = est
            
        self.fitted = True
        logger.info("Chronological stacking classifier fitted successfully.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using base models and pass them through the meta-model.
        """
        if not self.fitted:
            raise ModelingError("Stacking classifier must be fitted before prediction!")
            
        # Collect predictions from base models
        base_probs = {}
        for name, est in self.fitted_base_models.items():
            if hasattr(est, "predict_proba"):
                base_probs[name] = est.predict_proba(X)[:, 1]
            else:
                base_probs[name] = est.predict(X)
                
        X_meta = pd.DataFrame(base_probs)
        return self.meta_model.predict_proba(X_meta)[:, 1]
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)
