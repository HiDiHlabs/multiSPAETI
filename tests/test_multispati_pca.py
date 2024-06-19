import pytest
from sklearn.utils.estimator_checks import check_estimator

from multispaeti import MultispatiPCA


def test_sklearn_compatibility():
    try:
        check_estimator(MultispatiPCA())
    except AssertionError:
        pytest.fail("Failed check_estimator() test of sklearn.")
