from __future__ import annotations

import pytest
from package_submission import validate_student_id


@pytest.mark.parametrize("student_id", ["1234", "123456789012"])
def test_valid_student_id(student_id: str) -> None:
    assert validate_student_id(student_id) == student_id


@pytest.mark.parametrize("student_id", ["123", "1234567890123", "12a4", "１２３４", "../1234"])
def test_invalid_student_id(student_id: str) -> None:
    with pytest.raises(ValueError):
        validate_student_id(student_id)
