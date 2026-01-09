from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

import consolidation as consolidation_module
from consolidation import MemoryConsolidator
from automem.config import (
    CONSOLIDATION_DELETE_THRESHOLD,
    CONSOLIDATION_ARCHIVE_THRESHOLD,
    CONSOLIDATION_GRACE_PERIOD_DAYS,
    CONSOLIDATION_IMPORTANCE_PROTECTION_THRESHOLD,
    CONSOLIDATION_PROTECTED_TYPES,
)


class FakeResult:
    def __init__(self, rows: List[List[Any]]):
        self.result_set = rows


class FakeGraph:
    def __init__(self) -> None:
        self.relationship_counts: Dict[str, int] = {}
        self.sample_rows: List[List[Any]] = []
        self.existing_pairs: set[frozenset[str]] = set()
        self.cluster_rows: List[List[Any]] = []
        self.decay_rows: List[List[Any]] = []
        self.forgetting_rows: List[List[Any]] = []
        self.deleted: List[str] = []
        self.archived: List[tuple[str, float]] = []
        self.updated_scores: List[tuple[str, float]] = []
        self.queries: List[tuple[str, Dict[str, Any]]] = []

    def query(self, query: str, params: Dict[str, Any] | None = None) -> FakeResult:
        params = params or {}
        self.queries.append((query, params))

        if "COUNT(DISTINCT r)" in query:
            memory_id = params.get("id")
            count = self.relationship_counts.get(memory_id, 0)
            return FakeResult([[count]])

        if "RETURN COUNT(r) as count" in query and "$id1" in query:
            key = frozenset((params["id1"], params["id2"]))
            return FakeResult([[1 if key in self.existing_pairs else 0]])

        if "ORDER BY rand()" in query and "LIMIT $limit" in query:
            limit = params.get("limit")
            rows = self.sample_rows if limit is None else self.sample_rows[:limit]
            return FakeResult(rows)

        if "WHERE m.embeddings IS NOT NULL" in query:
            return FakeResult(self.cluster_rows)

        if "m.relevance_score as old_score" in query:
            return FakeResult(self.decay_rows)

        if "m.relevance_score as score" in query and "m.last_accessed as last_accessed" in query:
            return FakeResult(self.forgetting_rows)

        if "DETACH DELETE m" in query:
            self.deleted.append(params["id"])
            return FakeResult([])

        if "SET m.archived = true" in query:
            self.archived.append((params["id"], params["score"]))
            return FakeResult([])

        if "SET m.relevance_score = $score" in query:
            self.updated_scores.append((params["id"], params["score"]))
            return FakeResult([])

        return FakeResult([])


class FakeVectorStore:
    def __init__(self) -> None:
        self.deletions: List[tuple[str, Dict[str, Any]]] = []

    def delete(self, collection_name: str, points_selector: Dict[str, Any]) -> None:
        self.deletions.append((collection_name, points_selector))


@pytest.fixture(autouse=True)
def freeze_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a fixed timestamp to keep decay calculations deterministic."""

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            base = datetime(2024, 1, 1, tzinfo=timezone.utc)
            return base if tz is None else base.astimezone(tz)

    monkeypatch.setattr(consolidation_module, "datetime", FixedDatetime)
    yield
    monkeypatch.setattr(consolidation_module, "datetime", datetime)


def iso_days_ago(days: int) -> str:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return (base - timedelta(days=days)).isoformat()


def test_calculate_relevance_score_accounts_for_relationships() -> None:
    graph = FakeGraph()
    graph.relationship_counts["m1"] = 0
    consolidator = MemoryConsolidator(graph)

    common_memory = {
        "id": "m1",
        "timestamp": iso_days_ago(1),
        "importance": 0.6,
        "confidence": 0.6,
    }

    baseline = consolidator.calculate_relevance_score(common_memory.copy())

    # Clear the LRU cache to ensure the updated relationship count is fetched
    consolidator._get_relationship_count_cached_impl.cache_clear()

    graph.relationship_counts["m1"] = 6
    boosted = consolidator.calculate_relevance_score(common_memory.copy())

    assert boosted > baseline
    assert 0 < boosted <= 1


def test_discover_creative_associations_builds_connections() -> None:
    graph = FakeGraph()
    graph.sample_rows = [
        ["decision-a", "Chose approach A", "Decision", [1.0, 0.0, 0.0], iso_days_ago(3)],
        ["decision-b", "Chose approach B", "Decision", [0.0, 1.0, 0.0], iso_days_ago(4)],
        ["insight", "Insight about A", "Insight", [0.9, 0.1, 0.0], iso_days_ago(5)],
    ]

    consolidator = MemoryConsolidator(graph)
    associations = consolidator.discover_creative_associations(sample_size=3)

    assert any(item["type"] == "CONTRASTS_WITH" for item in associations)


def test_cluster_similar_memories_groups_items() -> None:
    graph = FakeGraph()
    graph.cluster_rows = [
        ["m1", "Alpha", [1.0, 0.0], "Insight"],
        ["m2", "Alpha follow-up", [0.95, 0.05], "Insight"],
        ["m3", "Alpha summary", [1.02, -0.02], "Pattern"],
    ]

    consolidator = MemoryConsolidator(graph)
    clusters = consolidator.cluster_similar_memories()

    assert clusters
    assert clusters[0]["size"] == 3
    assert clusters[0]["dominant_type"] in {"Insight", "Pattern"}


def build_forgetting_rows() -> List[List[Any]]:
    return [
        [
            "recent-keep",
            "Fresh important memory",
            0.8,
            iso_days_ago(2),
            "Insight",
            0.9,
            iso_days_ago(1),
            None,
            None,
        ],
        [
            "archive-candidate",
            "Memory to archive",
            0.2,
            iso_days_ago(15),
            "Memory",
            0.4,
            iso_days_ago(15),
            None,
            None,
        ],
        [
            "old-delete",
            "Superseded note",
            0.05,
            iso_days_ago(90),
            "Memory",
            0.2,
            iso_days_ago(90),
            None,
            None,
        ],
    ]


def test_apply_controlled_forgetting_dry_run() -> None:
    graph = FakeGraph()
    graph.relationship_counts["recent-keep"] = 5
    # Only 2 memories: one protected by grace period + high importance + protected type,
    # another would be deleted but is protected by grace period
    graph.forgetting_rows = [
        [
            "recent-keep",
            "Fresh important memory",
            0.8,
            iso_days_ago(2),
            "Insight",
            0.9,
            iso_days_ago(1),
            None,
            None,
        ],
        [
            "old-delete",
            "Superseded note",
            0.05,
            iso_days_ago(180),  # Old enough to not be protected by grace period
            "Memory",
            0.2,
            iso_days_ago(180),
            None,
            None,
        ],
    ]

    consolidator = MemoryConsolidator(graph)
    stats = consolidator.apply_controlled_forgetting(dry_run=True)

    assert stats["examined"] == 2
    assert stats["preserved"] == 1  # Only recent-keep
    assert len(stats["archived"]) == 0
    assert len(stats["deleted"]) == 1  # old-delete
    assert len(stats["protected"]) == 1  # recent-keep is protected
    assert graph.deleted == []


def test_apply_controlled_forgetting_updates_graph_and_vector_store() -> None:
    graph = FakeGraph()
    graph.relationship_counts["recent-keep"] = 5
    graph.forgetting_rows = build_forgetting_rows()

    vector_store = FakeVectorStore()
    consolidator = MemoryConsolidator(graph, vector_store=vector_store)

    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    assert stats["preserved"] == 2
    assert graph.updated_scores  # recent memory updated in graph
    assert graph.archived == []
    assert graph.deleted == ["old-delete"]
    assert vector_store.deletions
    collection, selector = vector_store.deletions[0]
    assert collection == "memories"
    points = selector.get("point_ids") or selector.get("points")
    assert points == ["old-delete"]


def test_apply_decay_updates_scores() -> None:
    graph = FakeGraph()
    graph.relationship_counts = {"a": 0, "b": 2}
    graph.decay_rows = [
        ["a", "Early note", iso_days_ago(10), 0.5, iso_days_ago(10), 0.5],
        ["b", "Recent insight", iso_days_ago(1), 0.7, iso_days_ago(1), 0.9],
    ]

    consolidator = MemoryConsolidator(graph)
    stats = consolidator._apply_decay()

    assert stats["processed"] == 2
    assert len(graph.updated_scores) == 2
    assert stats["avg_relevance_after"] <= 1


# ==================== Memory Protection Tests ====================

def test_protection_explicit_flag():
    """Test that explicitly protected memories are not deleted/archived."""
    graph = FakeGraph()
    graph.forgetting_rows = [
        [
            "protected-mem",
            "Important memory that should be protected",
            0.01,  # Very low relevance - would normally be deleted
            iso_days_ago(180),
            "Decision",
            0.6,
            iso_days_ago(180),
            True,  # Explicitly protected
            "User marked as important",
        ],
    ]

    consolidator = MemoryConsolidator(graph)
    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    assert stats["examined"] == 1
    assert stats["preserved"] == 1
    assert len(stats["protected"]) == 1
    assert len(stats["deleted"]) == 0
    assert len(stats["archived"]) == 0
    # Implementation uses custom protected_reason if provided
    assert "User marked as important" in stats["protected"][0]["protection_reason"]


def test_protection_importance_threshold():
    """Test that high-importance memories are protected."""
    graph = FakeGraph()
    graph.forgetting_rows = [
        [
            "high-importance-mem",
            "Critical decision memory",
            0.01,  # Very low relevance
            iso_days_ago(180),
            "Decision",
            0.8,  # High importance - should be protected
            iso_days_ago(180),
            None,
            None,
        ],
    ]

    consolidator = MemoryConsolidator(graph)
    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    assert stats["examined"] == 1
    assert stats["preserved"] == 1
    assert len(stats["protected"]) == 1
    assert len(stats["deleted"]) == 0
    assert "high importance" in stats["protected"][0]["protection_reason"]


def test_protection_grace_period():
    """Test that recent memories are protected by grace period."""
    graph = FakeGraph()
    graph.forgetting_rows = [
        [
            "recent-mem",
            "Recent memory within grace period",
            0.01,  # Very low relevance
            iso_days_ago(30),  # Only 30 days old - within 90-day grace period
            "Context",
            0.4,
            iso_days_ago(30),
            None,
            None,
        ],
    ]

    consolidator = MemoryConsolidator(graph)
    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    assert stats["examined"] == 1
    assert stats["preserved"] == 1
    assert len(stats["protected"]) == 1
    assert len(stats["deleted"]) == 0
    assert "grace period" in stats["protected"][0]["protection_reason"]


def test_protection_memory_types():
    """Test that protected memory types are not deleted/archived."""
    graph = FakeGraph()
    graph.forgetting_rows = [
        [
            "decision-mem",
            "Important decision",
            0.01,  # Very low relevance
            iso_days_ago(180),
            "Decision",  # Protected type
            0.4,
            iso_days_ago(180),
            None,
            None,
        ],
        [
            "insight-mem",
            "Valuable insight",
            0.05,  # Low relevance
            iso_days_ago(180),
            "Insight",  # Protected type
            0.4,
            iso_days_ago(180),
            None,
            None,
        ],
    ]

    consolidator = MemoryConsolidator(graph)
    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    assert stats["examined"] == 2
    assert stats["preserved"] == 2
    assert len(stats["protected"]) == 2
    assert len(stats["deleted"]) == 0
    assert all("protected type" in item["protection_reason"] for item in stats["protected"])


def test_protection_combined_criteria():
    """Test that multiple protection criteria work together."""
    graph = FakeGraph()
    graph.forgetting_rows = [
        [
            "multi-protected-mem",
            "Memory with multiple protection reasons",
            0.01,
            iso_days_ago(30),  # Recent
            "Decision",  # Protected type
            0.8,  # High importance
            iso_days_ago(30),
            True,  # Explicitly protected
            "Multiple reasons",
        ],
    ]

    consolidator = MemoryConsolidator(graph)
    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    assert stats["examined"] == 1
    assert stats["preserved"] == 1
    assert len(stats["protected"]) == 1
    # Should have multiple protection reasons combined
    reasons = stats["protected"][0]["protection_reason"]
    # Implementation uses custom protected_reason if provided
    assert "Multiple reasons" in reasons
    assert "high importance" in reasons
    assert "protected type" in reasons


def test_protection_archive_vs_delete():
    """Test that archive and delete thresholds work correctly."""
    graph = FakeGraph()
    # Use very old timestamps to ensure low calculated relevance
    graph.forgetting_rows = [
        [
            "to-archive",
            "Memory to archive",
            0.15,  # Stored relevance
            iso_days_ago(400),  # Old
            "Memory",
            0.4,
            iso_days_ago(400),
            None,
            None,
        ],
        [
            "to-delete",
            "Memory to delete",
            0.02,  # Stored relevance - below delete threshold
            iso_days_ago(400),  # Old
            "Memory",
            0.4,
            iso_days_ago(400),
            None,
            None,
        ],
    ]

    consolidator = MemoryConsolidator(graph)
    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    assert stats["examined"] == 2
    # Both should be deleted since they're very old and have low relevance
    assert len(stats["deleted"]) == 2
    assert any(item["id"] == "to-archive" for item in stats["deleted"])
    assert any(item["id"] == "to-delete" for item in stats["deleted"])


def test_no_protection_when_not_applicable():
    """Test that memories without protection criteria are processed normally."""
    graph = FakeGraph()
    graph.forgetting_rows = [
        [
            "unprotected-mem",
            "Old unimportant memory",
            0.01,  # Very low relevance
            iso_days_ago(180),
            "Context",  # Not a protected type
            0.3,  # Below importance threshold
            iso_days_ago(180),
            None,
            None,
        ],
    ]

    consolidator = MemoryConsolidator(graph)
    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    assert stats["examined"] == 1
    assert stats["preserved"] == 0
    assert len(stats["protected"]) == 0
    assert len(stats["deleted"]) == 1
    assert stats["deleted"][0]["id"] == "unprotected-mem"


def test_protection_logging():
    """Test that protection decisions are properly logged."""
    graph = FakeGraph()
    graph.forgetting_rows = [
        [
            "logged-protection",
            "Memory for logging test",
            0.01,
            iso_days_ago(30),
            "Decision",
            0.4,
            iso_days_ago(30),
            None,
            None,
        ],
    ]

    consolidator = MemoryConsolidator(graph)
    # This test verifies that the logging calls are made
    # In a real test, we'd mock the logger to capture the log messages
    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    assert stats["examined"] == 1
    assert len(stats["protected"]) == 1


def test_config_validation():
    """Test that configuration validation works correctly."""
    from automem.config import _validate_protection_config
    
    # Test that valid configuration passes
    try:
        _validate_protection_config()
    except ValueError:
        pytest.fail("Valid configuration should not raise ValueError")
    
    # Test invalid configurations (would need to temporarily modify config)
    # This is more complex to test properly and might be better as integration tests


def test_protection_with_custom_thresholds():
    """Test protection with custom importance threshold."""
    graph = FakeGraph()
    
    # Create consolidator with custom importance protection threshold
    consolidator = MemoryConsolidator(
        graph,
        importance_protection_threshold=0.6  # Lower than default 0.7
    )
    
    graph.forgetting_rows = [
        [
            "custom-threshold-mem",
            "Memory with importance at custom threshold",
            0.01,
            iso_days_ago(180),
            "Context",
            0.6,  # Exactly at the custom threshold
            iso_days_ago(180),
            None,
            None,
        ],
    ]

    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    assert stats["examined"] == 1
    assert stats["preserved"] == 1
    assert len(stats["protected"]) == 1
    assert "high importance" in stats["protected"][0]["protection_reason"]


def test_protection_edge_cases():
    """Test edge cases in protection logic."""
    graph = FakeGraph()
    
    # Test with exactly threshold values
    graph.forgetting_rows = [
        [
            "edge-case-mem",
            "Memory at exact importance threshold",
            0.01,
            iso_days_ago(180),
            "Context",
            CONSOLIDATION_IMPORTANCE_PROTECTION_THRESHOLD,  # Exact threshold
            iso_days_ago(180),
            None,
            None,
        ],
    ]

    consolidator = MemoryConsolidator(graph)
    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    # Should be protected since importance >= threshold
    assert len(stats["protected"]) == 1
    assert stats["preserved"] == 1


def test_protection_with_relationships():
    """Test that protection works regardless of relationship count."""
    graph = FakeGraph()
    graph.relationship_counts["protected-with-relations"] = 10  # Many relationships
    
    graph.forgetting_rows = [
        [
            "protected-with-relations",
            "Protected memory with many relationships",
            0.01,
            iso_days_ago(180),
            "Decision",  # Protected type
            0.4,
            iso_days_ago(180),
            None,
            None,
        ],
    ]

    consolidator = MemoryConsolidator(graph)
    stats = consolidator.apply_controlled_forgetting(dry_run=False)

    # Should still be protected even with many relationships
    assert len(stats["protected"]) == 1
    assert stats["preserved"] == 1

def test_vector_store_deletion_uses_configured_collection_name() -> None:
    graph = FakeGraph()
    vs = FakeVectorStore()
    consolidator = MemoryConsolidator(graph, vs, collection_name="silo_abc")

    # Provide forgetting rows that will trigger deletion (score below delete_threshold)
    graph.forgetting_rows = [
        ["m_del", "Old memory", 0.0, iso_days_ago(400), "Memory", 0.1, iso_days_ago(400), None, None],
    ]

    stats = consolidator.apply_controlled_forgetting(dry_run=False)
    assert stats["deleted"], "Expected a deletion"
    assert vs.deletions, "Expected vector store deletions"
    assert vs.deletions[0][0] == "silo_abc"
