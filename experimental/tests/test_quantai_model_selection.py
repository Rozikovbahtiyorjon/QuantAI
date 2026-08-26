from __future__ import annotations

import pytest

from experimental.src.quantai_model_selection import (
    ModelCandidateResult,
    ModelSelectionDecision,
    ModelSelectionResult,
    QuantAIModelSelectionEngine,
    select_champion_model,
)


def candidate(
    name: str,
    validation: float,
    test: float,
    stability: float = 1.0,
) -> ModelCandidateResult:
    return ModelCandidateResult(
        name=name,
        validation_score=validation,
        test_score=test,
        stability_score=stability,
    )


def test_all_candidates_are_ranked_and_best_is_selected():
    engine = QuantAIModelSelectionEngine()

    result = engine.select(
        [
            candidate(
                "model_a",
                0.80,
                0.70,
                0.80,
            ),
            candidate(
                "model_b",
                0.90,
                0.85,
                0.90,
            ),
            candidate(
                "model_c",
                0.70,
                0.75,
                0.70,
            ),
        ]
    )

    assert isinstance(
        result,
        ModelSelectionResult,
    )

    assert result.passed is True
    assert result.decision.selected is True
    assert result.champion_name == "model_b"
    assert result.total_candidates == 3


def test_performance_score_is_average_of_three_metrics():
    model = candidate(
        "model",
        0.8,
        0.7,
        0.9,
    )

    assert model.performance_score == pytest.approx(
        0.8
    )


def test_thresholds_filter_ineligible_candidates():
    engine = QuantAIModelSelectionEngine(
        minimum_validation_score=0.8,
        minimum_test_score=0.75,
        minimum_stability_score=0.8,
    )

    result = engine.select(
        [
            candidate(
                "weak",
                0.9,
                0.9,
                0.7,
            ),
            candidate(
                "champion",
                0.85,
                0.80,
                0.85,
            ),
        ]
    )

    assert result.passed is True
    assert result.champion_name == "champion"


def test_no_eligible_candidate_fails():
    engine = QuantAIModelSelectionEngine(
        minimum_validation_score=0.9,
        minimum_test_score=0.9,
        minimum_stability_score=0.9,
    )

    result = engine.select(
        [
            candidate(
                "weak",
                0.8,
                0.8,
                0.8,
            )
        ]
    )

    assert result.passed is False
    assert result.decision.selected is False
    assert result.champion_name is None
    assert result.errors


def test_current_champion_is_preserved_without_improvement():
    engine = QuantAIModelSelectionEngine(
        minimum_improvement=0.10
    )

    champion = candidate(
        "current",
        0.80,
        0.80,
        0.80,
    )

    challenger = candidate(
        "challenger",
        0.82,
        0.82,
        0.82,
    )

    result = engine.select(
        [challenger],
        current_champion=champion,
    )

    assert result.passed is True
    assert result.champion_name == "current"
    assert result.decision.selected is False
    assert result.warnings


def test_challenger_replaces_champion_after_required_improvement():
    engine = QuantAIModelSelectionEngine(
        minimum_improvement=0.01
    )

    champion = candidate(
        "current",
        0.70,
        0.70,
        0.70,
    )

    challenger = candidate(
        "challenger",
        0.90,
        0.90,
        0.90,
    )

    result = engine.select(
        [challenger],
        current_champion=champion,
    )

    assert result.passed is True
    assert result.champion_name == "challenger"
    assert result.decision.selected is True


def test_below_threshold_current_champion_is_replaced():
    engine = QuantAIModelSelectionEngine(
        minimum_validation_score=0.8,
        minimum_test_score=0.8,
        minimum_stability_score=0.8,
    )

    champion = candidate(
        "old",
        0.70,
        0.70,
        0.70,
    )

    challenger = candidate(
        "new",
        0.81,
        0.82,
        0.83,
    )

    result = engine.select(
        [challenger],
        current_champion=champion,
    )

    assert result.champion_name == "new"
    assert result.passed is True


def test_challengers_are_returned():
    engine = QuantAIModelSelectionEngine()

    result = engine.select(
        [
            candidate(
                "a",
                0.70,
                0.70,
                0.70,
            ),
            candidate(
                "b",
                0.90,
                0.90,
                0.90,
            ),
            candidate(
                "c",
                0.80,
                0.80,
                0.80,
            ),
        ]
    )

    assert [
        item.name
        for item in result.decision.challengers
    ] == [
        "c",
        "a",
    ]


def test_evaluator_supports_external_model_objects():
    engine = QuantAIModelSelectionEngine()

    models = [
        "A",
        "B",
    ]

    def evaluator(
        model: str,
    ) -> ModelCandidateResult:
        if model == "A":
            return candidate(
                "A",
                0.7,
                0.7,
                0.7,
            )

        return candidate(
            "B",
            0.8,
            0.8,
            0.8,
        )

    result = engine.select_with_evaluator(
        models,
        evaluator,
    )

    assert result.passed is True
    assert result.champion_name == "B"


def test_convenience_function_selects_champion():
    result = select_champion_model(
        [
            candidate(
                "A",
                0.7,
                0.7,
                0.7,
            ),
            candidate(
                "B",
                0.8,
                0.8,
                0.8,
            ),
        ]
    )

    assert result.passed is True
    assert result.champion_name == "B"


def test_empty_candidates_are_rejected():
    engine = QuantAIModelSelectionEngine()

    with pytest.raises(ValueError):
        engine.select([])


def test_non_iterable_candidates_are_rejected():
    engine = QuantAIModelSelectionEngine()

    with pytest.raises(TypeError):
        engine.select(None)


def test_invalid_candidate_type_is_rejected():
    engine = QuantAIModelSelectionEngine()

    with pytest.raises(TypeError):
        engine.select(
            ["model"]
        )


def test_duplicate_candidate_names_are_rejected():
    engine = QuantAIModelSelectionEngine()

    with pytest.raises(ValueError):
        engine.select(
            [
                candidate(
                    "same",
                    0.8,
                    0.8,
                ),
                candidate(
                    "same",
                    0.9,
                    0.9,
                ),
            ]
        )


def test_invalid_scores_are_rejected():
    engine = QuantAIModelSelectionEngine()

    with pytest.raises(ValueError):
        engine.select(
            [
                candidate(
                    "bad",
                    1.1,
                    0.8,
                )
            ]
        )

    with pytest.raises(ValueError):
        QuantAIModelSelectionEngine(
            minimum_test_score=-0.1
        )


def test_boolean_score_is_rejected():
    with pytest.raises(TypeError):
        QuantAIModelSelectionEngine().select(
            [
                ModelCandidateResult(
                    name="bad",
                    validation_score=True,
                    test_score=0.8,
                )
            ]
        )


def test_invalid_evaluator_is_rejected():
    with pytest.raises(TypeError):
        QuantAIModelSelectionEngine().select_with_evaluator(
            ["A"],
            None,
        )


def test_invalid_current_champion_is_rejected():
    with pytest.raises(TypeError):
        QuantAIModelSelectionEngine().select(
            [
                candidate(
                    "A",
                    0.8,
                    0.8,
                )
            ],
            current_champion="A",
        )


def test_decision_dataclass_contract():
    decision = ModelSelectionDecision(
        champion=None,
        challengers=(),
        selected=False,
        reason="test",
    )

    assert decision.selected is False
    assert decision.reason == "test"