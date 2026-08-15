import pytest

from src.champion_evaluator import ChampionEvaluator, EvaluationResult


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


def test_default_weights_are_normalized():
    evaluator = ChampionEvaluator()

    assert sum(evaluator.weights.values()) == pytest.approx(1.0)


def test_evaluation_returns_expected_type():
    evaluator = ChampionEvaluator()

    result = evaluator.evaluate(metrics(), metrics())

    assert isinstance(result, EvaluationResult)


def test_equal_strategies_are_not_qualified():
    evaluator = ChampionEvaluator()

    result = evaluator.evaluate(metrics(), metrics())

    assert result.qualified is False
    assert result.improvement == pytest.approx(0.0)


def test_better_candidate_is_qualified():
    evaluator = ChampionEvaluator()

    candidate = metrics(
        profit_factor=2.0,
        net_profit=150.0,
        win_rate=0.60,
        sharpe_ratio=1.30,
        max_drawdown=8.0,
    )

    result = evaluator.evaluate(candidate, metrics())

    assert result.qualified is True
    assert result.improvement > 0


def test_worse_candidate_is_rejected():
    evaluator = ChampionEvaluator()

    candidate = metrics(
        profit_factor=1.2,
        net_profit=70.0,
        win_rate=0.50,
        sharpe_ratio=0.7,
        max_drawdown=15.0,
    )

    result = evaluator.evaluate(candidate, metrics())

    assert result.qualified is False
    assert result.improvement < 0


def test_lower_drawdown_improves_candidate_score():
    evaluator = ChampionEvaluator()

    candidate = metrics(max_drawdown=5.0)
    champion = metrics(max_drawdown=10.0)

    result = evaluator.evaluate(candidate, champion)

    assert result.metrics["max_drawdown"] > 0


def test_higher_drawdown_reduces_candidate_score():
    evaluator = ChampionEvaluator()

    candidate = metrics(max_drawdown=15.0)
    champion = metrics(max_drawdown=10.0)

    result = evaluator.evaluate(candidate, champion)

    assert result.metrics["max_drawdown"] < 0


def test_minimum_improvement_threshold():
    evaluator = ChampionEvaluator(min_improvement=0.5)

    candidate = metrics(
        profit_factor=1.55,
        net_profit=105.0,
        win_rate=0.56,
        sharpe_ratio=1.05,
        max_drawdown=9.5,
    )

    result = evaluator.evaluate(candidate, metrics())

    assert result.qualified is False


def test_compare_returns_boolean():
    evaluator = ChampionEvaluator()

    assert evaluator.compare(metrics(), metrics()) is False


def test_missing_metric_is_rejected():
    evaluator = ChampionEvaluator()

    invalid = metrics()
    invalid.pop("sharpe_ratio")

    with pytest.raises(ValueError):
        evaluator.evaluate(invalid, metrics())


def test_extra_metric_is_rejected():
    evaluator = ChampionEvaluator()

    invalid = metrics()
    invalid["extra"] = 1.0

    with pytest.raises(ValueError):
        evaluator.evaluate(invalid, metrics())


def test_non_numeric_metric_is_rejected():
    evaluator = ChampionEvaluator()

    invalid = metrics()
    invalid["net_profit"] = "100"

    with pytest.raises(TypeError):
        evaluator.evaluate(invalid, metrics())