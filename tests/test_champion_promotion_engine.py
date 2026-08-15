import pytest

from src.champion_evaluator import ChampionEvaluator
from src.champion_promotion_engine import (
    ChampionPromotionEngine,
    PromotionResult,
)


def metrics(
    profit_factor=1.5,
    net_profit=100.0,
    win_rate=0.55,
    sharpe_ratio=1.0,
    max_drawdown=10.0,
):
    return {
        "profit_factor": profit_factor,
        "net_profit": net_profit,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


def test_default_engine_uses_evaluator():
    engine = ChampionPromotionEngine()

    assert isinstance(engine.evaluator, ChampionEvaluator)


def test_evaluate_returns_promotion_result():
    engine = ChampionPromotionEngine()

    result = engine.evaluate(metrics(), metrics())

    assert isinstance(result, PromotionResult)


def test_equal_candidate_is_rejected():
    engine = ChampionPromotionEngine()

    result = engine.evaluate(metrics(), metrics())

    assert result.promoted is False
    assert result.reason == "candidate_does_not_outperform_champion"


def test_better_candidate_is_promoted():
    engine = ChampionPromotionEngine()

    candidate = metrics(
        profit_factor=2.0,
        net_profit=150.0,
        win_rate=0.60,
        sharpe_ratio=1.30,
        max_drawdown=8.0,
    )

    result = engine.evaluate(candidate, metrics())

    assert result.promoted is True
    assert result.reason == "candidate_outperforms_champion"


def test_worse_candidate_is_rejected():
    engine = ChampionPromotionEngine()

    candidate = metrics(
        profit_factor=1.2,
        net_profit=70.0,
        win_rate=0.50,
        sharpe_ratio=0.7,
        max_drawdown=15.0,
    )

    result = engine.evaluate(candidate, metrics())

    assert result.promoted is False


def test_should_promote_matches_evaluation():
    engine = ChampionPromotionEngine()

    candidate = metrics(
        profit_factor=2.0,
        net_profit=150.0,
        win_rate=0.60,
        sharpe_ratio=1.30,
        max_drawdown=8.0,
    )

    result = engine.evaluate(candidate, metrics())

    assert engine.should_promote(candidate, metrics()) is result.promoted


def test_custom_evaluator_is_used():
    evaluator = ChampionEvaluator(min_improvement=0.5)
    engine = ChampionPromotionEngine(evaluator=evaluator)

    candidate = metrics(
        profit_factor=1.55,
        net_profit=105.0,
        win_rate=0.56,
        sharpe_ratio=1.05,
        max_drawdown=9.5,
    )

    result = engine.evaluate(candidate, metrics())

    assert result.promoted is False


def test_evaluation_is_preserved():
    engine = ChampionPromotionEngine()

    result = engine.evaluate(metrics(), metrics())

    assert result.evaluation.candidate_score == pytest.approx(0.0)
    assert result.evaluation.improvement == pytest.approx(0.0)


def test_missing_metric_raises_error():
    engine = ChampionPromotionEngine()

    candidate = metrics()
    candidate.pop("sharpe_ratio")

    with pytest.raises(ValueError):
        engine.evaluate(candidate, metrics())


def test_extra_metric_raises_error():
    engine = ChampionPromotionEngine()

    candidate = metrics()
    candidate["extra"] = 1.0

    with pytest.raises(ValueError):
        engine.evaluate(candidate, metrics())


def test_invalid_metric_type_raises_error():
    engine = ChampionPromotionEngine()

    candidate = metrics()
    candidate["net_profit"] = "100"

    with pytest.raises(TypeError):
        engine.evaluate(candidate, metrics())