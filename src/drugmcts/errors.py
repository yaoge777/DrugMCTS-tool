class DrugMCTSError(Exception):
    """An actionable tool error, safe to show without credentials or remote response bodies."""


class ConfigurationError(DrugMCTSError):
    pass


class InputError(DrugMCTSError):
    pass


class ModelError(DrugMCTSError):
    pass


class BudgetExceeded(ModelError):
    pass
