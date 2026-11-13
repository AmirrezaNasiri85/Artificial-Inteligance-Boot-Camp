import pytest

from scripts.package_submission import validate_student_id


@pytest.mark.parametrize("value", ["1234", "123456789012"])
def test_valid_id(value):
    assert validate_student_id(value) == value


@pytest.mark.parametrize("value", ["123", "1234567890123", "12a4", "１２３４", "../1234"])
def test_invalid_id(value):
    with pytest.raises(ValueError):
        validate_student_id(value)
