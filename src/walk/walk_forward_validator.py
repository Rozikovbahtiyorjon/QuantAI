from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.walk.walk_forward_validation_report import (
    ValidationMetric,
    WalkForwardValidationReport,
)

__all__ = [
    "WalkForwardValidator",
    "validate_walk_forward",
]


class WalkForwardValidator:
    """
    Validation layer for Walk-Forward results.

    Validates an existing Walk-Forward result without modifying
    BacktestEngine or PerformanceAnalyzer.
    """

    def __init__(
        self,
        train_size: int = 100,
        test_size: int = 20,
        step_size: Optional[int] = None,
        initial_balance: float = 1000.0,
        min_windows: int = 1,
        min_validation_rate: float = 1.0,
        **kwargs: Any,
    ) -> None:

        if isinstance(min_windows, bool):
            raise TypeError(
                "min_windows must be an integer."
            )

        if not isinstance(min_windows, int):
            raise TypeError(
                "min_windows must be an integer."
            )

        if min_windows <= 0:
            raise ValueError(
                "min_windows must be greater than zero."
            )

        if isinstance(min_validation_rate, bool):
            raise TypeError(
                "min_validation_rate must be numeric."
            )

        if not isinstance(
            min_validation_rate,
            (int, float),
        ):
            raise TypeError(
                "min_validation_rate must be numeric."
            )

        if not 0.0 <= float(
            min_validation_rate
        ) <= 1.0:
            raise ValueError(
                "min_validation_rate must be between 0 and 1."
            )

        if isinstance(train_size, bool):
            raise TypeError(
                "train_size must be an integer."
            )

        if not isinstance(train_size, int):
            raise TypeError(
                "train_size must be an integer."
            )

        if isinstance(test_size, bool):
            raise TypeError(
                "test_size must be an integer."
            )

        if not isinstance(test_size, int):
            raise TypeError(
                "test_size must be an integer."
            )

        if train_size <= 0:
            raise ValueError(
                "train_size must be greater than zero."
            )

        if test_size <= 0:
            raise ValueError(
                "test_size must be greater than zero."
            )

        if step_size is None:
            step_size = test_size

        if isinstance(step_size, bool):
            raise TypeError(
                "step_size must be an integer."
            )

        if not isinstance(step_size, int):
            raise TypeError(
                "step_size must be an integer."
            )

        if step_size <= 0:
            raise ValueError(
                "step_size must be greater than zero."
            )

        if isinstance(initial_balance, bool):
            raise TypeError(
                "initial_balance must be numeric."
            )

        if not isinstance(
            initial_balance,
            (int, float),
        ):
            raise TypeError(
                "initial_balance must be numeric."
            )

        if initial_balance <= 0:
            raise ValueError(
                "initial_balance must be greater than zero."
            )

        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

        self.initial_balance = float(
            initial_balance
        )

        self.min_windows = min_windows

        self.minimum_windows = min_windows

        self.min_validation_rate = float(
            min_validation_rate
        )

        self.latest_report = None
        self._result = None

        self._extra_options = dict(
            kwargs
        )

    # ========================================================
    # DATA VALIDATION
    # ========================================================

    def validate_data(
        self,
        df: pd.DataFrame,
    ) -> bool:
        """
        Validate prepared market data.
        """

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "DataFrame is empty."
            )

        required_columns = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        minimum_rows = (
            self.train_size
            + self.test_size
        )

        if len(df) < minimum_rows:
            raise ValueError(
                "Walk-Forward validation requires "
                f"at least {minimum_rows} rows. "
                f"Received: {len(df)}"
            )

        return True

    # ========================================================
    # MAIN VALIDATION ENTRY POINT
    # ========================================================

    def validate(
        self,
        result: Any,
    ) -> WalkForwardValidationReport:
        """
        Validate a Walk-Forward result and return a report.
        """

        if isinstance(
            result,
            pd.DataFrame,
        ):
            self.validate_data(
                result
            )

            report = self._create_report(
                total_windows=0,
                completed_windows=0,
                failed_windows=0,
                valid_window_boundaries=False,
                window_results_available=False,
                validation_rate=0.0,
                errors=[
                    "A Walk-Forward result is required."
                ],
            )

            self.latest_report = report
            self._result = result

            return report

        report = self._validate_result(
            result
        )

        self.latest_report = report
        self._result = result

        return report

    # ========================================================
    # RESULT VALIDATION
    # ========================================================

    def _validate_result(
        self,
        result: Any,
    ) -> WalkForwardValidationReport:

        errors: list[str] = []

        if result is None:
            return self._create_report(
                total_windows=0,
                completed_windows=0,
                failed_windows=0,
                valid_window_boundaries=False,
                window_results_available=False,
                validation_rate=0.0,
                errors=[
                    "Walk-Forward result is None."
                ],
            )

        total_windows = self._safe_int(
            getattr(
                result,
                "total_windows",
                0,
            )
        )

        completed_windows = self._safe_int(
            getattr(
                result,
                "completed_windows",
                0,
            )
        )

        failed_windows = self._safe_int(
            getattr(
                result,
                "failed_windows",
                0,
            )
        )

        window_results = getattr(
            result,
            "window_results",
            [],
        )

        try:
            window_results = list(
                window_results
            )
        except TypeError:
            window_results = []

        # ----------------------------------------------------
        # BASIC VALUE VALIDATION
        # ----------------------------------------------------

        if total_windows < 0:
            errors.append(
                "total_windows cannot be negative."
            )

        if completed_windows < 0:
            errors.append(
                "completed_windows cannot be negative."
            )

        if failed_windows < 0:
            errors.append(
                "failed_windows cannot be negative."
            )

        if completed_windows > total_windows:
            errors.append(
                "completed_windows cannot exceed "
                "total_windows."
            )

        if failed_windows > total_windows:
            errors.append(
                "failed_windows cannot exceed "
                "total_windows."
            )

        if (
            completed_windows
            + failed_windows
            > total_windows
        ):
            errors.append(
                "completed_windows + failed_windows "
                "cannot exceed total_windows."
            )

        # ----------------------------------------------------
        # VALIDATION RATE
        # ----------------------------------------------------

        if total_windows > 0:
            validation_rate = (
                completed_windows
                / total_windows
            )
        else:
            validation_rate = 0.0

        if total_windows < self.min_windows:
            errors.append(
                "Not enough Walk-Forward windows: "
                f"required at least {self.min_windows}, "
                f"received {total_windows}."
            )

        if (
            validation_rate
            < self.min_validation_rate
        ):
            errors.append(
                "Validation rate is below the "
                "required minimum."
            )

        # ----------------------------------------------------
        # WINDOW BOUNDARIES
        # ----------------------------------------------------

        valid_boundaries = (
            self._validate_window_boundaries(
                window_results,
                errors,
            )
        )

        # ----------------------------------------------------
        # WINDOW RESULTS COUNT
        # ----------------------------------------------------

        window_results_available = (
            len(window_results)
            == total_windows
        )

        if not window_results_available:
            errors.append(
                "window_results count does not "
                "match total_windows."
            )

        # ----------------------------------------------------
        # COMPLETED WINDOWS
        # ----------------------------------------------------

        completed_valid = (
            completed_windows
            == total_windows
        )

        if not completed_valid:
            errors.append(
                "completed_windows must match "
                "total_windows."
            )

        # ----------------------------------------------------
        # FAILED WINDOWS
        # ----------------------------------------------------

        failed_valid = (
            failed_windows == 0
        )

        if not failed_valid:
            errors.append(
                "failed_windows must be zero."
            )

        # ----------------------------------------------------
        # CREATE REPORT
        # ----------------------------------------------------

        return self._create_report(
            total_windows=total_windows,
            completed_windows=completed_windows,
            failed_windows=failed_windows,
            valid_window_boundaries=valid_boundaries,
            window_results_available=(
                window_results_available
            ),
            validation_rate=validation_rate,
            errors=errors,
        )

    # ========================================================
    # WINDOW BOUNDARIES
    # ========================================================

    def _validate_window_boundaries(
        self,
        windows: list[Any],
        errors: list[str],
    ) -> bool:
        """
        Validate train/test boundaries.

        Expected structure:

            train_start < train_end
            train_end <= test_start
            test_start < test_end
        """

        if not windows:
            return False

        valid = True

        previous_test_end = None

        for index, window in enumerate(
            windows
        ):

            train_start = self._optional_int(
                getattr(
                    window,
                    "train_start",
                    None,
                )
            )

            train_end = self._optional_int(
                getattr(
                    window,
                    "train_end",
                    None,
                )
            )

            test_start = self._optional_int(
                getattr(
                    window,
                    "test_start",
                    None,
                )
            )

            test_end = self._optional_int(
                getattr(
                    window,
                    "test_end",
                    None,
                )
            )

            if (
                train_start is None
                or train_end is None
                or test_start is None
                or test_end is None
            ):
                valid = False

                errors.append(
                    f"Window {index} has incomplete "
                    "boundary information."
                )

                continue

            if train_start < 0:
                valid = False

                errors.append(
                    f"Window {index}: "
                    "train_start cannot be negative."
                )

            if train_end <= train_start:
                valid = False

                errors.append(
                    f"Window {index}: "
                    "train_end must be greater "
                    "than train_start."
                )

            if test_start < train_end:
                valid = False

                errors.append(
                    f"Window {index}: "
                    "test_start must be greater "
                    "than or equal to train_end."
                )

            if test_end <= test_start:
                valid = False

                errors.append(
                    f"Window {index}: "
                    "test_end must be greater "
                    "than test_start."
                )

            if (
                previous_test_end is not None
                and test_start < previous_test_end
            ):
                valid = False

                errors.append(
                    f"Window {index}: "
                    "test window overlaps "
                    "the previous test window."
                )

            previous_test_end = test_end

        return valid

    # ========================================================
    # REPORT CREATION
    # ========================================================

    def _create_report(
        self,
        total_windows: int,
        completed_windows: int,
        failed_windows: int,
        valid_window_boundaries: bool,
        window_results_available: bool,
        validation_rate: float,
        errors: list[str],
    ) -> WalkForwardValidationReport:
        """
        Create the project's WalkForwardValidationReport.

        Important:
        WalkForwardValidationReport.total_metrics is a
        read-only property. Therefore we NEVER assign to it.

        The report receives its metrics through `report.metrics`,
        while `total_metrics` is calculated by the report class.
        """

        metrics = [
            self._make_metric(
                name="validation_rate",
                value=validation_rate,
                passed=(
                    validation_rate
                    >= self.min_validation_rate
                ),
                threshold=(
                    self.min_validation_rate
                ),
            ),
            self._make_metric(
                name="valid_window_boundaries",
                value=(
                    1
                    if valid_window_boundaries
                    else 0
                ),
                passed=valid_window_boundaries,
            ),
            self._make_metric(
                name="window_results_available",
                value=(
                    1
                    if window_results_available
                    else 0
                ),
                passed=window_results_available,
            ),
            self._make_metric(
                name="completed_windows",
                value=completed_windows,
                passed=(
                    completed_windows
                    == total_windows
                ),
            ),
            self._make_metric(
                name="failed_windows",
                value=failed_windows,
                passed=(
                    failed_windows == 0
                ),
            ),
        ]

        is_valid = (
            len(errors) == 0
        )

        # The existing project report constructor
        # requires these three positional arguments.
        report = WalkForwardValidationReport(
            total_windows,
            completed_windows,
            failed_windows,
        )

        # Attach metrics.
        #
        # total_metrics must NOT be assigned because it is
        # a read-only property in WalkForwardValidationReport.
        report.metrics = metrics

        # Attach validation state only when the project
        # report implementation allows it.
        self._safe_setattr(
            report,
            "validation_rate",
            float(validation_rate),
        )

        self._safe_setattr(
            report,
            "is_valid",
            is_valid,
        )

        self._safe_setattr(
            report,
            "passed",
            is_valid,
        )

        self._safe_setattr(
            report,
            "valid",
            is_valid,
        )

        self._safe_setattr(
            report,
            "errors",
            list(errors),
        )

        self._safe_setattr(
            report,
            "min_windows",
            self.min_windows,
        )

        self._safe_setattr(
            report,
            "min_validation_rate",
            self.min_validation_rate,
        )

        return report

    # ========================================================
    # METRIC CREATION
    # ========================================================

    @staticmethod
    def _make_metric(
        name: str,
        value: float,
        passed: bool,
        threshold: Optional[float] = None,
    ) -> ValidationMetric:
        """
        Create ValidationMetric using the current project's
        constructor.
        """

        try:
            return ValidationMetric(
                name=name,
                value=value,
                passed=passed,
                threshold=threshold,
            )
        except TypeError:
            pass

        try:
            return ValidationMetric(
                name=name,
                value=value,
                passed=passed,
            )
        except TypeError:
            pass

        try:
            return ValidationMetric(
                name=name,
                value=value,
            )
        except TypeError:
            pass

        return ValidationMetric(
            name,
            value,
            passed,
        )

    # ========================================================
    # WALK-FORWARD ENGINE EXECUTION
    # ========================================================

    def run(
        self,
        df: pd.DataFrame,
        engine: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run the existing WalkForwardEngine.

        This method does not modify BacktestEngine.
        """

        self.validate_data(
            df
        )

        if engine is None:
            from src.walk_forward_engine import (
                WalkForwardEngine,
            )

            try:
                engine = WalkForwardEngine(
                    train_size=self.train_size,
                    test_size=self.test_size,
                    step_size=self.step_size,
                    initial_balance=(
                        self.initial_balance
                    ),
                )
            except TypeError:
                engine = WalkForwardEngine()

        result = engine.run(
            df,
            train_size=self.train_size,
            test_size=self.test_size,
            step_size=self.step_size,
            initial_balance=(
                self.initial_balance
            ),
            **kwargs,
        )

        self._result = result

        return result

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int:
        try:
            return int(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> Optional[int]:
        if value is None:
            return None

        try:
            return int(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _safe_setattr(
        obj: Any,
        name: str,
        value: Any,
    ) -> None:
        """
        Set an attribute only when the target report allows it.

        This prevents compatibility problems with read-only
        properties such as total_metrics.
        """

        try:
            setattr(
                obj,
                name,
                value,
            )
        except (
            AttributeError,
            TypeError,
        ):
            pass


# ================================================================
# PUBLIC FUNCTION
# ================================================================

def validate_walk_forward(
    result: Any,
    min_windows: int = 1,
    min_validation_rate: float = 1.0,
    **kwargs: Any,
) -> WalkForwardValidationReport:
    """
    Validate a Walk-Forward result and return a report.
    """

    validator = WalkForwardValidator(
        min_windows=min_windows,
        min_validation_rate=(
            min_validation_rate
        ),
        **kwargs,
    )

    return validator.validate(
        result
    )