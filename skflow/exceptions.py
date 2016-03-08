__all__ = ['NotFittedError']


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backword compatibility.

    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import NotFittedError
    >>> try:
    ...     LinearSVC().predict([[1, 2,], [2, 4], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    NotFittedError('This Linear SVC instance is not fitted yet',)
    """
