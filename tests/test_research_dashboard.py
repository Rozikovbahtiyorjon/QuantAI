import pytest

from src.research_dashboard import (
    DashboardSection,
    ResearchDashboard,
    ResearchDashboardSnapshot,
)


def test_section_creation() -> None:
    section = DashboardSection(
        name="risk",
        status="HEALTHY",
        score=0.95,
        message="OK",
    )

    assert section.name == "risk"
    assert section.status == "HEALTHY"
    assert section.score == 0.95


def test_section_validation() -> None:
    with pytest.raises(ValueError):
        DashboardSection("", "HEALTHY")

    with pytest.raises(ValueError):
        DashboardSection("risk", "BAD")

    with pytest.raises(TypeError):
        DashboardSection(
            "risk",
            "HEALTHY",
            score="bad",
        )

    with pytest.raises(ValueError):
        DashboardSection(
            "risk",
            "HEALTHY",
            score=1.1,
        )


def test_default_sections() -> None:
    dashboard = ResearchDashboard()

    assert "market_regime" in dashboard.section_names
    assert "risk" in dashboard.section_names
    assert "paper_trading" in dashboard.section_names
    assert "reliability" in dashboard.section_names


def test_custom_sections() -> None:
    dashboard = ResearchDashboard(
        ["risk", "strategy"]
    )

    assert dashboard.section_names == (
        "risk",
        "strategy",
    )


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        ResearchDashboard([])

    with pytest.raises(ValueError):
        ResearchDashboard(
            ["risk", "risk"]
        )

    with pytest.raises(ValueError):
        ResearchDashboard([""])


def test_build_healthy_snapshot() -> None:
    dashboard = ResearchDashboard()

    snapshot = dashboard.build_snapshot(
        {
            "risk": DashboardSection(
                "risk",
                "HEALTHY",
            ),
            "strategy": DashboardSection(
                "strategy",
                "HEALTHY",
            ),
        },
        {
            "pnl": 125.5,
            "drawdown": 0.04,
        },
    )

    assert isinstance(
        snapshot,
        ResearchDashboardSnapshot,
    )

    assert snapshot.overall_status == "HEALTHY"
    assert snapshot.metrics["pnl"] == 125.5


def test_build_degraded_snapshot() -> None:
    dashboard = ResearchDashboard()

    snapshot = dashboard.build_snapshot(
        {
            "risk": DashboardSection(
                "risk",
                "HEALTHY",
            ),
            "monitoring": DashboardSection(
                "monitoring",
                "DEGRADED",
            ),
        }
    )

    assert snapshot.overall_status == "DEGRADED"


def test_build_critical_snapshot() -> None:
    dashboard = ResearchDashboard()

    snapshot = dashboard.build_snapshot(
        {
            "risk": DashboardSection(
                "risk",
                "CRITICAL",
            ),
            "strategy": DashboardSection(
                "strategy",
                "HEALTHY",
            ),
        }
    )

    assert snapshot.overall_status == "CRITICAL"


def test_build_unknown_snapshot() -> None:
    dashboard = ResearchDashboard()

    snapshot = dashboard.build_snapshot(
        {
            "risk": DashboardSection(
                "risk",
                "UNKNOWN",
            ),
        }
    )

    assert snapshot.overall_status == "UNKNOWN"


def test_build_from_statuses() -> None:
    dashboard = ResearchDashboard()

    snapshot = dashboard.build_from_statuses(
        {
            "risk": "HEALTHY",
            "strategy": "HEALTHY",
            "monitoring": "DEGRADED",
        }
    )

    assert snapshot.overall_status == "DEGRADED"
    assert len(snapshot.sections) == 3


def test_unknown_section_validation() -> None:
    dashboard = ResearchDashboard()

    with pytest.raises(ValueError):
        dashboard.build_from_statuses(
            {
                "unknown_section": "HEALTHY"
            }
        )


def test_snapshot_validation() -> None:
    section = DashboardSection(
        "risk",
        "HEALTHY",
    )

    with pytest.raises(ValueError):
        ResearchDashboardSnapshot(
            "BAD",
            (section,),
        )

    with pytest.raises(TypeError):
        ResearchDashboardSnapshot(
            "HEALTHY",
            [section],
        )

    with pytest.raises(TypeError):
        ResearchDashboardSnapshot(
            "HEALTHY",
            (section,),
            {"pnl": "bad"},
        )


def test_build_validation() -> None:
    dashboard = ResearchDashboard()

    with pytest.raises(TypeError):
        dashboard.build_snapshot("invalid")

    with pytest.raises(ValueError):
        dashboard.build_snapshot({})

    with pytest.raises(TypeError):
        dashboard.build_snapshot(
            {
                "risk": "invalid"
            }
        )

    with pytest.raises(TypeError):
        dashboard.build_snapshot(
            {
                "risk": DashboardSection(
                    "risk",
                    "HEALTHY",
                )
            },
            {"pnl": "bad"},
        )


def test_summary() -> None:
    dashboard = ResearchDashboard()

    snapshot = dashboard.build_snapshot(
        {
            "risk": DashboardSection(
                "risk",
                "HEALTHY",
            ),
            "strategy": DashboardSection(
                "strategy",
                "DEGRADED",
            ),
            "monitoring": DashboardSection(
                "monitoring",
                "CRITICAL",
            ),
        }
    )

    summary = dashboard.summarize(snapshot)

    assert summary["overall_status"] == "CRITICAL"
    assert summary["total_sections"] == 3
    assert summary["healthy_sections"] == 1
    assert summary["degraded_sections"] == 1
    assert summary["critical_sections"] == 1

    assert summary["health_ratio"] == pytest.approx(
        1 / 3
    )


def test_to_dict() -> None:
    dashboard = ResearchDashboard()

    snapshot = dashboard.build_snapshot(
        {
            "risk": DashboardSection(
                "risk",
                "HEALTHY",
                score=0.9,
            )
        },
        {
            "pnl": 100.0
        },
    )

    payload = dashboard.to_dict(snapshot)

    assert payload["overall_status"] == "HEALTHY"
    assert payload["metrics"]["pnl"] == 100.0
    assert payload["sections"][0]["name"] == "risk"
    assert payload["sections"][0]["score"] == 0.9
    assert "timestamp" in payload


def test_summary_validation() -> None:
    with pytest.raises(TypeError):
        ResearchDashboard.summarize("invalid")

    with pytest.raises(TypeError):
        ResearchDashboard.to_dict("invalid")