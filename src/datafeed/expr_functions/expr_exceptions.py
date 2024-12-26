class ExpressionError(Exception):
    """Base class for expression calculation errors."""
    pass

class InvalidPeriodError(ExpressionError):
    """Raised when a period parameter is invalid."""
    pass

class InvalidInputError(ExpressionError):
    """Raised when input data is invalid."""
    pass

class CalculationError(ExpressionError):
    """Raised when calculation fails."""
    pass 