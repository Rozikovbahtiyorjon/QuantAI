from __future__ import annotations

import py_compile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    message: str = ""


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    checks: tuple[ValidationCheck, ...]
    total_checks: int
    passed_checks: int
    failed_checks: int

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def failed_names(self) -> tuple[str, ...]:
        return tuple(
            check.name
            for check in self.checks
            if not check.passed
        )


class QuantAIEndToEndValidationEngine:
    """Standalone end-to-end and regression validation coordinator."""

    def __init__(
        self,
        project_root: str | Path = ".",
        required_modules: Iterable[str] | None = None,
    ) -> None:
        self.project_root = Path(project_root)

        if not self.project_root.exists():
            raise ValueError(
                "project_root does not exist."
            )

        if not self.project_root.is_dir():
            raise ValueError(
                "project_root must be a directory."
            )

        modules = (
            tuple(required_modules)
            if required_modules is not None
            else ()
        )

        for module in modules:
            if not isinstance(module, str):
                raise TypeError(
                    "required_modules entries must be strings."
                )

            if not module.strip():
                raise ValueError(
                    "required module names cannot be empty."
                )

        self.required_modules = modules

    def validate_required_modules(self) -> ValidationCheck:
        missing = []

        for module in self.required_modules:
            path = self._module_path(module)

            if not path.is_file():
                missing.append(module)

        if missing:
            return ValidationCheck(
                name="required_modules",
                passed=False,
                message=(
                    "Missing modules: "
                    + ", ".join(missing)
                ),
            )

        return ValidationCheck(
            name="required_modules",
            passed=True,
            message="All required modules are present.",
        )

    def validate_required_tests(self) -> ValidationCheck:
        missing = []

        for module in self.required_modules:
            test_path = self._test_path(module)

            if not test_path.is_file():
                missing.append(module)

        if missing:
            return ValidationCheck(
                name="required_tests",
                passed=False,
                message=(
                    "Missing tests: "
                    + ", ".join(missing)
                ),
            )

        return ValidationCheck(
            name="required_tests",
            passed=True,
            message="All required test files are present.",
        )

    def validate_python_syntax(
        self,
        directories: Iterable[str] = ("src", "tests"),
    ) -> ValidationCheck:
        errors = []

        for directory in directories:
            path = self.project_root / directory

            if not path.exists():
                continue

            if not path.is_dir():
                errors.append(
                    f"{directory}: not a directory"
                )
                continue

            for file_path in sorted(path.rglob("*.py")):
                try:
                    py_compile.compile(
                        str(file_path),
                        doraise=True,
                    )
                except py_compile.PyCompileError as exc:
                    errors.append(
                        f"{file_path}: {exc.msg}"
                    )

        if errors:
            return ValidationCheck(
                name="python_syntax",
                passed=False,
                message="; ".join(errors),
            )

        return ValidationCheck(
            name="python_syntax",
            passed=True,
            message="Python syntax validation passed.",
        )

    def validate_regression_status(
        self,
        results: Mapping[str, bool],
    ) -> ValidationCheck:
        if not isinstance(results, Mapping):
            raise TypeError(
                "results must be a mapping."
            )

        if not results:
            raise ValueError(
                "results cannot be empty."
            )

        for name, passed in results.items():
            if not isinstance(name, str) or not name.strip():
                raise TypeError(
                    "Regression check names must be "
                    "non-empty strings."
                )

            if not isinstance(passed, bool):
                raise TypeError(
                    "Regression check results must be booleans."
                )

        failed = [
            name
            for name, passed in results.items()
            if not passed
        ]

        if failed:
            return ValidationCheck(
                name="regression_status",
                passed=False,
                message=(
                    "Failed regression checks: "
                    + ", ".join(failed)
                ),
            )

        return ValidationCheck(
            name="regression_status",
            passed=True,
            message="All supplied regression checks passed.",
        )

    def validate(
        self,
        regression_results: Mapping[str, bool] | None = None,
    ) -> ValidationReport:
        checks = [
            self.validate_required_modules(),
            self.validate_required_tests(),
            self.validate_python_syntax(),
        ]

        if regression_results is not None:
            checks.append(
                self.validate_regression_status(
                    regression_results
                )
            )

        passed_checks = sum(
            check.passed
            for check in checks
        )

        failed_checks = (
            len(checks) - passed_checks
        )

        return ValidationReport(
            passed=failed_checks == 0,
            checks=tuple(checks),
            total_checks=len(checks),
            passed_checks=passed_checks,
            failed_checks=failed_checks,
        )

    def _module_path(self, module: str) -> Path:
        normalized = module.replace(".", "/")

        if not normalized.endswith(".py"):
            normalized += ".py"

        return (
            self.project_root
            / "src"
            / normalized
        )

    def _test_path(self, module: str) -> Path:
        normalized = module.replace(".", "/")

        if normalized.endswith(".py"):
            normalized = normalized[:-3]

        return (
            self.project_root
            / "tests"
            / f"test_{normalized}.py"
        )