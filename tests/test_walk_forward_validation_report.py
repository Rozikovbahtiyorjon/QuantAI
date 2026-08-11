from src.walk.walk_forward_validation_report import (
    ValidationMetric,
    WalkForwardValidationReport,
)


def test_validation_metric_creation() -> None:
    metric = ValidationMetric(
        name="test",
        value=1.0,
        passed=True,
    )

    assert metric.name == "test"
    assert metric.value == 1.0
    assert metric.passed is True


def test_report_starts_empty() -> None:
    report = WalkForwardValidationReport(
        total_windows=3,
        completed_windows=3,
        failed_windows=0,
    )

    assert report.total_metrics == 0
    assert report.passed_metrics == 0
    assert report.failed_metrics == 0
    assert report.validation_rate == 0.0
    assert report.passed is False


def test_add_metric() -> None:
    report = WalkForwardValidationReport(
        total_windows=1,
        completed_windows=1,
        failed_windows=0,
    )

    report.add_metric(
        name="example",
        value=1.0,
        passed=True,
    )

    assert report.total_metrics == 1
    assert report.passed_metrics == 1
    assert report.failed_metrics == 0
    assert report.validation_rate == 1.0


def test_failed_metric() -> None:
    report = WalkForwardValidationReport(
        total_windows=1,
        completed_windows=1,
        failed_windows=0,
    )

    report.add_metric(
        name="example",
        value=0.0,
        passed=False,
    )

    assert report.total_metrics == 1
    assert report.passed_metrics == 0
    assert report.failed_metrics == 1
    assert report.validation_rate == 0.0


def test_finalize_passes_when_all_checks_pass() -> None:
    report = WalkForwardValidationReport(
        total_windows=2,
        completed_windows=2,
        failed_windows=0,
    )

    report.add_metric(
        name="metric_1",
        value=1.0,
        passed=True,
    )

    report.finalize()

    assert report.passed is True


def test_finalize_fails_when_window_failed() -> None:
    report = WalkForwardValidationReport(
        total_windows=2,
        completed_windows=1,
        failed_windows=1,
    )

    report.add_metric(
        name="metric_1",
        value=1.0,
        passed=True,
    )

    report.finalize()

    assert report.passed is False


def test_finalize_fails_when_metric_failed() -> None:
    report = WalkForwardValidationReport(
        total_windows=2,
        completed_windows=2,
        failed_windows=0,
    )

    report.add_metric(
        name="metric_1",
        value=0.0,
        passed=False,
    )

    report.finalize()

    assert report.passed is False


def test_to_dict_contains_expected_fields() -> None:
    report = WalkForwardValidationReport(
        total_windows=2,
        completed_windows=2,
        failed_windows=0,
    )

    report.add_metric(
        name="metric_1",
        value=1.0,
        passed=True,
        threshold=1.0,
        description="test",
    )

    report.finalize()

    data = report.to_dict()

    assert data["total_windows"] == 2
    assert data["completed_windows"] == 2
    assert data["failed_windows"] == 0
    assert data["total_metrics"] == 1
    assert data["passed_metrics"] == 1
    assert data["failed_metrics"] == 0
    assert data["validation_rate"] == 1.0
    assert data["passed"] is True
    assert len(data["metrics"]) == 1