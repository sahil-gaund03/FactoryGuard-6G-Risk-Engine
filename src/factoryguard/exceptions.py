class FactoryGuardError(Exception):
    """Base exception class for all errors in FactoryGuard 6G."""
    pass

class DataValidationError(FactoryGuardError):
    """Raised when the dataset fails schema validation or data quality checks."""
    pass

class FeatureEngineeringError(FactoryGuardError):
    """Raised when there is an error calculating rolling, lag, trend or baseline features."""
    pass

class ModelingError(FactoryGuardError):
    """Raised when a model fails to train, calibrate, evaluate or make predictions."""
    pass

class ConfigurationError(FactoryGuardError):
    """Raised when configuration files are missing, malformed, or invalid."""
    pass
