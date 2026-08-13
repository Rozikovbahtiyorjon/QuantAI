import pytest

from src.quantai_reliability import (
    QuantAIReliability,
    ReliabilityAction,
    ReliabilityCheck,
    ReliabilityReport,
    ReliabilityStatus,
)


def test_healthy_report() -> None:
    engine = QuantAIReliability()

    report = engine.evaluate(
        [
            ReliabilityCheck("data", True),
            ReliabilityCheck("model", True),
            ReliabilityCheck("risk", True),
            ReliabilityCheck("execution", True),
        ]
    )

    assert report.status is ReliabilityStatus.HEALTHY
    assert report.action is ReliabilityAction.CONTINUE
    assert report.score == 1.0
    assert engine.can_continue(report) is True


def test_degraded_report() -> None:
    engine = QuantAIReliability()

    report = engine.evaluate(
        [
            ReliabilityCheck("data", True),
            ReliabilityCheck("model", True),
            ReliabilityCheck("risk", True),
            ReliabilityCheck(
                "execution",
                False,
                severity="WARNING",
            ),
        ]
    )

    assert report.status is ReliabilityStatus.DEGRADED
    assert report.action is ReliabilityAction.DEGRADE
    assert report.score == 0.75
    assert engine.can_continue(report) is False


def test_critical_report() -> None:
    engine = QuantAIReliability()

    report = engine.evaluate(
        [
            ReliabilityCheck("data", True),
            ReliabilityCheck(
                "exchange",
                False,
                severity="CRITICAL",
            ),
        ]
    )

    assert report.status is ReliabilityStatus.CRITICAL
    assert report.action is ReliabilityAction.HALT
    assert report.score == 0.5


def test_mapping_evaluation() -> None:
    report = QuantAIReliability().evaluate_mapping(
        {
            "data": True,
            "model": True,
            "risk": True,
            "execution": True,
        }
    )

    assert report.status is ReliabilityStatus.HEALTHY
    assert len(report.checks) == 4


def test_recovery_report() -> None:
    report = QuantAIReliability().recovery_report()

    assert report.status is ReliabilityStatus.HEALTHY
    assert report.action is ReliabilityAction.RECOVER
    assert report.score == 1.0


def test_report_to_dict() -> None:
    report = QuantAIReliability().evaluate_mapping(
        {"data": True}
    )

    payload = report.to_dict()

    assert payload["status"] == "HEALTHY"
    assert payload["action"] == "CONTINUE"
    assert payload["score"] == 1.0
    assert payload["checks"][0]["name"] == "data"
    assert "timestamp" in payload


def test_check_validation() -> None:
    with pytest.raises(ValueError):
        ReliabilityCheck("", True)

    with pytest.raises(TypeError):
        ReliabilityCheck("data", "true")

    with pytest.raises(ValueError):
        ReliabilityCheck(
            "data",
            True,
            severity="BAD",
        )

    with pytest.raises(TypeError):
        ReliabilityCheck(
            "data",
            True,
            message=1,
        )


def test_report_validation() -> None:
    check = ReliabilityCheck(
        "data",
        True,
    )

    with pytest.raises(ValueError):
        ReliabilityReport(
            ReliabilityStatus.HEALTHY,
            ReliabilityAction.CONTINUE,
            1.1,
            (check,),
        )

    with pytest.raises(TypeError):
        ReliabilityReport(
            ReliabilityStatus.HEALTHY,
            ReliabilityAction.CONTINUE,
            1.0,
            [check],
        )


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        QuantAIReliability(
            {
                "healthy": 0.7,
                "degraded": 0.8,
            }
        )

    with pytest.raises(ValueError):
        QuantAIReliability(
            {
                "healthy": 1.2,
                "degraded": 0.5,
            }
        )

    with pytest.raises(TypeError):
        QuantAIReliability(
            {
                "healthy": "high",
                "degraded": 0.5,
            }
        )

    with pytest.raises(ValueError):
        QuantAIReliability(
            {
                "healthy": 0.9,
            }
        )


def test_evaluate_validation() -> None:
    engine = QuantAIReliability()

    with pytest.raises(TypeError):
        engine.evaluate("invalid")

    with pytest.raises(ValueError):
        engine.evaluate([])

    with pytest.raises(TypeError):
        engine.evaluate(
            [
                ReliabilityCheck("data", True),
                "invalid",
            ]
        )


def test_mapping_validation() -> None:
    engine = QuantAIReliability()

    with pytest.raises(TypeError):
        engine.evaluate_mapping("invalid")

    with pytest.raises(ValueError):
        engine.evaluate_mapping({})

    with pytest.raises(ValueError):
        engine.evaluate_mapping({"": True})

    with pytest.raises(TypeError):
        engine.evaluate_mapping({"data": 1})


def test_can_continue_validation() -> None:
    with pytest.raises(TypeError):
        QuantAIReliability().can_continue("invalid")


def test_custom_thresholds() -> None:
    engine = QuantAIReliability(
        {
            "healthy": 0.8,
            "degraded": 0.5,
        }
    )

    report = engine.evaluate(
        [
            ReliabilityCheck("a", True),
            ReliabilityCheck("b", True),
            ReliabilityCheck("c", True),
            ReliabilityCheck("d", True),
            ReliabilityCheck(
                "e",
                False,
                severity="WARNING",
            ),
        ]
    )

    assert report.status is ReliabilityStatus.HEALTHY

    assert engine.thresholds == {
        "healthy": 0.8,
        "degraded": 0.5,
    }