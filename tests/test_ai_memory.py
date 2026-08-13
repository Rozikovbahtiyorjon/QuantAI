import pytest

from src.ai_memory import AIMemory, MEMORY_LAYERS, MemoryRecord


def test_memory_layers() -> None:
    assert MEMORY_LAYERS == (
        "permanent",
        "strategy",
        "asset",
        "regime",
        "experiment",
    )


def test_add_and_get() -> None:
    memory = AIMemory()

    record = memory.add(
        memory_id="m1",
        layer="strategy",
        content=(
            "Trend strategy performs well in strong momentum."
        ),
        tags=["trend", "momentum", "trend"],
        strategy_id="s1",
        asset="BTCUSDT",
        regime="TREND_UP",
        confidence=0.9,
        importance=0.8,
    )

    assert record.memory_id == "m1"
    assert record.tags == ("trend", "momentum")
    assert memory.get("m1") == record
    assert memory.size == 1


def test_score() -> None:
    record = MemoryRecord(
        memory_id="m1",
        layer="permanent",
        content="validated rule",
        confidence=1.0,
        importance=0.5,
    )

    assert record.score == 0.8


def test_duplicate_memory_is_rejected() -> None:
    memory = AIMemory()

    memory.add(
        memory_id="m1",
        layer="experiment",
        content="experiment",
    )

    with pytest.raises(ValueError):
        memory.add(
            memory_id="m1",
            layer="experiment",
            content="duplicate",
        )


def test_search_by_query_and_filters() -> None:
    memory = AIMemory()

    memory.add(
        memory_id="trend-btc",
        layer="strategy",
        content="BTC trend strategy has stable results",
        tags=["trend", "stable"],
        strategy_id="s1",
        asset="BTCUSDT",
        regime="TREND_UP",
        confidence=0.9,
        importance=0.9,
    )

    memory.add(
        memory_id="range-eth",
        layer="strategy",
        content="ETH range strategy",
        tags=["range"],
        strategy_id="s2",
        asset="ETHUSDT",
        regime="RANGE",
        confidence=0.7,
        importance=0.7,
    )

    results = memory.search(
        "BTC trend",
        layer="strategy",
        asset="BTCUSDT",
    )

    assert [item.memory_id for item in results] == [
        "trend-btc"
    ]


def test_search_by_tags() -> None:
    memory = AIMemory()

    memory.add(
        memory_id="m1",
        layer="regime",
        content="high volatility observation",
        tags=["volatility", "shock"],
    )

    memory.add(
        memory_id="m2",
        layer="regime",
        content="low volatility observation",
        tags=["volatility", "calm"],
    )

    results = memory.search(
        tags=["volatility", "shock"]
    )

    assert [item.memory_id for item in results] == ["m1"]


def test_search_is_sorted_by_score() -> None:
    memory = AIMemory()

    memory.add(
        memory_id="low",
        layer="experiment",
        content="same keyword finding",
        confidence=0.2,
        importance=0.2,
    )

    memory.add(
        memory_id="high",
        layer="experiment",
        content="same keyword finding",
        confidence=1.0,
        importance=1.0,
    )

    results = memory.search("same keyword")

    assert [item.memory_id for item in results] == [
        "high",
        "low",
    ]


def test_limit_and_best() -> None:
    memory = AIMemory()

    for index in range(3):
        memory.add(
            memory_id=f"m{index}",
            layer="experiment",
            content="research observation",
            confidence=0.5 + index * 0.1,
            importance=0.5,
        )

    assert len(
        memory.search(
            "research observation",
            limit=2,
        )
    ) == 2

    assert (
        memory.best(
            "research observation"
        ).memory_id
        == "m2"
    )


def test_archive_and_default_visibility() -> None:
    memory = AIMemory()

    memory.add(
        memory_id="m1",
        layer="strategy",
        content="archivable finding",
    )

    archived = memory.archive("m1")

    assert archived.status == "archived"
    assert memory.search("archivable finding") == []

    assert (
        memory.search(
            "archivable finding",
            include_archived=True,
        )[0].memory_id
        == "m1"
    )


def test_remove_and_clear() -> None:
    memory = AIMemory()

    memory.add(
        memory_id="m1",
        layer="experiment",
        content="finding",
    )

    assert memory.remove("m1") is True
    assert memory.remove("m1") is False
    assert memory.size == 0

    memory.add(
        memory_id="m2",
        layer="experiment",
        content="finding",
    )

    memory.clear()

    assert memory.size == 0


def test_export() -> None:
    memory = AIMemory()

    memory.add(
        memory_id="m1",
        layer="asset",
        content="BTC observation",
        tags=["btc"],
        asset="BTCUSDT",
    )

    exported = memory.export()

    assert len(exported) == 1
    assert exported[0]["memory_id"] == "m1"
    assert exported[0]["tags"] == ["btc"]
    assert exported[0]["asset"] == "BTCUSDT"
    assert isinstance(
        exported[0]["created_at"],
        str,
    )


def test_record_validation() -> None:
    with pytest.raises(ValueError):
        MemoryRecord(
            memory_id="",
            layer="strategy",
            content="x",
        )

    with pytest.raises(ValueError):
        MemoryRecord(
            memory_id="m",
            layer="invalid",
            content="x",
        )

    with pytest.raises(ValueError):
        MemoryRecord(
            memory_id="m",
            layer="strategy",
            content="",
        )

    with pytest.raises(ValueError):
        MemoryRecord(
            memory_id="m",
            layer="strategy",
            content="x",
            confidence=1.1,
        )

    with pytest.raises(TypeError):
        MemoryRecord(
            memory_id="m",
            layer="strategy",
            content="x",
            tags=["tag"],
        )


def test_search_validation() -> None:
    memory = AIMemory()

    with pytest.raises(TypeError):
        memory.search(123)

    with pytest.raises(ValueError):
        memory.search(layer="invalid")

    with pytest.raises(TypeError):
        memory.search(limit="2")

    with pytest.raises(ValueError):
        memory.search(limit=0)


def test_archive_unknown_memory() -> None:
    memory = AIMemory()

    with pytest.raises(KeyError):
        memory.archive("missing")


def test_empty_search() -> None:
    memory = AIMemory()

    assert memory.search() == []
    assert memory.best() is None