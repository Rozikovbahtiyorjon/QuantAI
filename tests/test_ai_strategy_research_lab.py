import pytest

from src.ai_strategy_research_lab import (
    AIStrategyResearchLaboratory,
    ResearchCandidate,
    ResearchEvidence,
)


def make_evidence(
    value: float = 0.8,
) -> ResearchEvidence:
    return ResearchEvidence(
        backtest_score=value,
        walk_forward_score=value,
        robustness_score=value,
        monte_carlo_score=value,
        stress_score=value,
    )


def test_candidate_validation() -> None:
    with pytest.raises(TypeError):
        ResearchCandidate(123)

    with pytest.raises(ValueError):
        ResearchCandidate("")

    with pytest.raises(TypeError):
        ResearchCandidate(
            "strategy",
            parameters="invalid",
        )


def test_evidence_validation() -> None:
    with pytest.raises(TypeError):
        ResearchEvidence(
            "bad",
            0.8,
            0.8,
            0.8,
            0.8,
        )

    with pytest.raises(ValueError):
        ResearchEvidence(
            1.1,
            0.8,
            0.8,
            0.8,
            0.8,
        )


def test_constructor_validation() -> None:
    with pytest.raises(TypeError):
        AIStrategyResearchLaboratory("invalid")

    with pytest.raises(TypeError):
        AIStrategyResearchLaboratory(
            lambda _: make_evidence(),
            "bad",
        )

    with pytest.raises(ValueError):
        AIStrategyResearchLaboratory(
            lambda _: make_evidence(),
            1.1,
        )


def test_accepts_strong_candidate() -> None:
    lab = AIStrategyResearchLaboratory(
        lambda _: make_evidence(0.9)
    )

    result = lab.evaluate(
        ResearchCandidate("strategy_a")
    )

    assert result.accepted is True
    assert result.research_score == pytest.approx(0.9)
    assert result.rejection_reason is None


def test_rejects_weak_candidate() -> None:
    lab = AIStrategyResearchLaboratory(
        lambda _: make_evidence(0.5)
    )

    result = lab.evaluate(
        ResearchCandidate("strategy_b")
    )

    assert result.accepted is False
    assert result.research_score == pytest.approx(0.5)
    assert result.rejection_reason is not None


def test_weighted_score() -> None:
    lab = AIStrategyResearchLaboratory(
        lambda _: ResearchEvidence(
            backtest_score=1.0,
            walk_forward_score=0.8,
            robustness_score=0.6,
            monte_carlo_score=0.4,
            stress_score=0.2,
        )
    )

    result = lab.evaluate(
        ResearchCandidate("strategy_c")
    )

    expected = (
        0.20 * 1.0
        + 0.25 * 0.8
        + 0.20 * 0.6
        + 0.15 * 0.4
        + 0.20 * 0.2
    )

    assert result.research_score == pytest.approx(
        expected
    )


def test_evaluator_result_type_is_validated() -> None:
    lab = AIStrategyResearchLaboratory(
        lambda _: "invalid"
    )

    with pytest.raises(TypeError):
        lab.evaluate(
            ResearchCandidate("strategy_d")
        )


def test_candidate_type_is_validated() -> None:
    lab = AIStrategyResearchLaboratory(
        lambda _: make_evidence()
    )

    with pytest.raises(TypeError):
        lab.evaluate("invalid")


def test_evaluate_many() -> None:
    lab = AIStrategyResearchLaboratory(
        lambda _: make_evidence(0.8)
    )

    results = lab.evaluate_many(
        [
            ResearchCandidate("a"),
            ResearchCandidate("b"),
            ResearchCandidate("c"),
        ]
    )

    assert len(results) == 3
    assert all(
        result.accepted
        for result in results
    )


def test_evaluate_many_rejects_strings() -> None:
    lab = AIStrategyResearchLaboratory(
        lambda _: make_evidence()
    )

    with pytest.raises(TypeError):
        lab.evaluate_many("strategy")


def test_evaluator_receives_candidate() -> None:
    received = []

    def evaluator(
        candidate: ResearchCandidate,
    ) -> ResearchEvidence:
        received.append(candidate.strategy_id)
        return make_evidence()

    lab = AIStrategyResearchLaboratory(
        evaluator
    )

    lab.evaluate(
        ResearchCandidate("strategy_x")
    )

    assert received == ["strategy_x"]


def test_boundary_threshold_accepts_equal_score() -> None:
    lab = AIStrategyResearchLaboratory(
        lambda _: make_evidence(0.7),
        acceptance_threshold=0.7,
    )

    result = lab.evaluate(
        ResearchCandidate("strategy_y")
    )

    assert result.accepted is True


def test_parameters_are_preserved() -> None:
    candidate = ResearchCandidate(
        "strategy_z",
        parameters={
            "period": 20,
            "threshold": 0.65,
        },
    )

    assert candidate.parameters["period"] == 20
    assert candidate.parameters["threshold"] == 0.65