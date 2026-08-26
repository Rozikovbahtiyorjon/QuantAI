from pathlib import Path

import pytest

from experimental.src.quantai_end_to_end_validation import (
    QuantAIEndToEndValidationEngine,
)


def make_project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()

    (tmp_path / "src" / "alpha.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    (tmp_path / "tests" / "test_alpha.py").write_text(
        "def test_alpha():\n"
        "    assert True\n",
        encoding="utf-8",
    )

    return tmp_path


def test_constructor_defaults() -> None:
    engine = QuantAIEndToEndValidationEngine()

    assert engine.required_modules == ()


def test_constructor_accepts_path() -> None:
    engine = QuantAIEndToEndValidationEngine(
        Path(".")
    )

    assert engine.project_root.is_dir()


def test_constructor_rejects_missing_root() -> None:
    with pytest.raises(ValueError):
        QuantAIEndToEndValidationEngine(
            "missing-project-root"
        )


def test_constructor_rejects_file_root(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "project.txt"

    file_path.write_text(
        "x",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        QuantAIEndToEndValidationEngine(
            file_path
        )


def test_constructor_rejects_invalid_module_type(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        QuantAIEndToEndValidationEngine(
            tmp_path,
            required_modules=[123],
        )


def test_constructor_rejects_empty_module(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        QuantAIEndToEndValidationEngine(
            tmp_path,
            required_modules=[""],
        )


def test_required_modules_pass(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path)

    engine = QuantAIEndToEndValidationEngine(
        root,
        required_modules=["alpha"],
    )

    check = engine.validate_required_modules()

    assert check.passed is True


def test_required_modules_fail(
    tmp_path: Path,
) -> None:
    engine = QuantAIEndToEndValidationEngine(
        tmp_path,
        required_modules=["missing"],
    )

    check = engine.validate_required_modules()

    assert check.passed is False
    assert "missing" in check.message


def test_required_tests_pass(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path)

    engine = QuantAIEndToEndValidationEngine(
        root,
        required_modules=["alpha"],
    )

    check = engine.validate_required_tests()

    assert check.passed is True


def test_required_tests_fail(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path)

    test_file = (
        root
        / "tests"
        / "test_alpha.py"
    )

    test_file.unlink()

    engine = QuantAIEndToEndValidationEngine(
        root,
        required_modules=["alpha"],
    )

    check = engine.validate_required_tests()

    assert check.passed is False


def test_syntax_validation_pass(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path)

    engine = QuantAIEndToEndValidationEngine(
        root
    )

    check = engine.validate_python_syntax()

    assert check.passed is True


def test_syntax_validation_fail(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path)

    broken_file = (
        root
        / "src"
        / "broken.py"
    )

    broken_file.write_text(
        "def broken(:\n",
        encoding="utf-8",
    )

    engine = QuantAIEndToEndValidationEngine(
        root
    )

    check = engine.validate_python_syntax()

    assert check.passed is False


def test_regression_status_pass() -> None:
    engine = QuantAIEndToEndValidationEngine()

    check = engine.validate_regression_status(
        {
            "core": True,
            "risk": True,
        }
    )

    assert check.passed is True


def test_regression_status_fail() -> None:
    engine = QuantAIEndToEndValidationEngine()

    check = engine.validate_regression_status(
        {
            "core": True,
            "risk": False,
        }
    )

    assert check.passed is False
    assert "risk" in check.message


def test_regression_status_validation() -> None:
    engine = QuantAIEndToEndValidationEngine()

    with pytest.raises(TypeError):
        engine.validate_regression_status(
            "invalid"
        )

    with pytest.raises(ValueError):
        engine.validate_regression_status({})

    with pytest.raises(TypeError):
        engine.validate_regression_status(
            {
                "core": 1,
            }
        )

    with pytest.raises(TypeError):
        engine.validate_regression_status(
            {
                "": True,
            }
        )


def test_full_validation_report(
    tmp_path: Path,
) -> None:
    root = make_project(tmp_path)

    engine = QuantAIEndToEndValidationEngine(
        root,
        required_modules=["alpha"],
    )

    report = engine.validate(
        {
            "feature_engine": True,
            "risk_engine": True,
        }
    )

    assert report.passed is True
    assert report.status == "PASS"
    assert report.total_checks == 4
    assert report.passed_checks == 4
    assert report.failed_checks == 0
    assert report.failed_names() == ()