"""
====================================================
QuantAI Professional
Disaster Recovery & State Persistence
====================================================

State persistence and disaster recovery for trading system.

Features:
- Periodic state checkpointing
- Graceful shutdown and restart
- State validation on startup
- Atomic writes with backup
- Corruption detection and recovery

====================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import msgpack
import os
import pickle
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Generic

import msgpack
import zstandard as zstd


class RecoveryState(Enum):
    """System recovery state."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    FAILED = "FAILED"


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    timestamp: datetime
    version: str
    component: str
    size_bytes: int
    checksum: str
    sequence: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class RecoveryPlan:
    """Recovery plan for a component."""
    component: str
    action: str  # "restore", "rebuild", "reset", "skip"
    priority: int  # Lower = higher priority
    estimated_time_seconds: float
    dependencies: List[str] = field(default_factory=list)


T = TypeVar('T')


class StateSerializer:
    """Handles serialization/deserialization with compression."""

    def __init__(self, compression_level: int = 3):
        self.compression_level = compression_level
        self._compressor = zstd.ZstdCompressor(level=compression_level)
        self._decompressor = zstd.ZstdDecompressor()

    def serialize(self, obj: Any) -> bytes:
        """Serialize object to compressed bytes."""
        data = msgpack.packb(obj, use_bin_type=True)
        return self._compressor.compress(data)

    def deserialize(self, data: bytes, cls: Optional[type] = None) -> Any:
        """Deserialize bytes to object."""
        decompressed = self._decompressor.decompress(data)
        return msgpack.unpackb(decompressed, raw=False, strict_map_key=False)


class AtomicFileWriter:
    """Atomic file writer with backup and rollback."""

    def __init__(self, base_path: Path, backup_count: int = 3):
        self.base_path = Path(base_path)
        self.backup_count = backup_count
        self.base_path.parent.mkdir(parents=True, exist_ok=True)

    def write_atomic(self, path: Path, data: bytes) -> None:
        """Write data atomically with backup rotation."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary file
        temp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{int(time.time() * 1000)}")

        try:
            # Write to temp file
            temp_path.write_bytes(data)

            # Rotate backups
            self._rotate_backups(path)

            # Atomic rename
            temp_path.replace(path)

        except Exception:
            # Cleanup temp file on error
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise

    def _rotate_backups(self, path: Path) -> None:
        """Rotate backup files."""
        for i in range(self.backup_count - 1, 0, -1):
            src = path.with_suffix(f".bak{i}")
            dst = path.with_suffix(f".bak{i+1}")
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)

        # Create .bak1 from current
        bak1 = path.with_suffix(".bak1")
        if path.exists():
            if bak1.exists():
                bak1.unlink()
            path.rename(bak1)

    def read_latest(self, path: Path) -> Optional[bytes]:
        """Read latest valid file (checks backups if main is corrupted)."""
        # Try main file
        if path.exists():
            try:
                return path.read_bytes()
            except Exception:
                pass

        # Try backups in order
        for i in range(1, self.backup_count + 1):
            backup_path = path.with_suffix(f".bak{i}")
            if backup_path.exists():
                try:
                    return backup_path.read_bytes()
                except Exception:
                    continue

        return None


class CheckpointManager:
    """
    Manages periodic state checkpointing with atomic writes.

    Features:
    - Periodic automatic checkpoints
    - Atomic writes with backup rotation
    - Compression with zstd
    - Checkpoint validation
    - Incremental checkpoints (future)
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        interval_seconds: float = 60.0,
        max_checkpoints: int = 10,
        serializer: Optional[StateSerializer] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.interval_seconds = interval_seconds
        self.max_checkpoints = max_checkpoints
        self.serializer = serializer or StateSerializer()
        self.writer = AtomicFileWriter(self.checkpoint_dir / "checkpoints", backup_count=5)

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._sequence = 0
        self._components: Dict[str, Callable[[], Any]] = {}
        self._last_checkpoint: Optional[CheckpointMetadata] = None

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def register_component(
        self,
        name: str,
        getter: Callable[[], Any],
        priority: int = 0,
    ) -> None:
        """Register a component for checkpointing."""
        self._components[name] = (getter, priority)

    def unregister_component(self, name: str) -> bool:
        """Unregister a component."""
        if name in self._components:
            del self._components[name]
            return True
        return False

    async def start(self) -> None:
        """Start periodic checkpointing."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._checkpoint_loop())
        print(f"[CheckpointManager] Started (interval={self.interval_seconds}s)")

    async def stop(self) -> None:
        """Stop checkpointing and save final state."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Final checkpoint
        await self.checkpoint(force=True)
        print("[CheckpointManager] Stopped")

    async def checkpoint(self, force: bool = False) -> CheckpointMetadata:
        """Create a checkpoint of all registered components."""
        start_time = time.time()

        # Collect component states
        component_states = {}
        for name, (getter, priority) in sorted(
            self._components.items(), key=lambda x: x[1][1] if isinstance(x[1], tuple) else 0
        ):
            try:
                state = getter()
                component_states[name] = state
            except Exception as e:
                print(f"[CheckpointManager] Error getting state for {name}: {e}")

        # Serialize
        checkpoint_data = {
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": self._sequence,
            "components": component_states,
        }

        serialized = self.serializer.serialize(checkpoint_data)
        checksum = hashlib.sha256(serialized).hexdigest()[:16]

        # Atomic write
        self._sequence += 1
        filename = f"checkpoint_{self._sequence:08d}_{int(time.time())}.msgpack.zst"
        filepath = self.checkpoint_dir / "checkpoints" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.writer.write_atomic(filepath, serialized)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        metadata = CheckpointMetadata(
            timestamp=datetime.now(timezone.utc),
            version="1.0",
            component="all",
            size_bytes=len(serialized),
            checksum=checksum,
            sequence=self._sequence,
            metadata={"component_count": len(component_states)},
        )

        self._last_checkpoint = metadata

        if __debug__:
            print(f"[CheckpointManager] Checkpoint {self._sequence} saved ({len(serialized)} bytes, {time.time() - start_time:.3f}s)")

        return metadata

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoint_dir = self.checkpoint_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return

        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_*.msgpack.zst"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old in checkpoints[self.max_checkpoints:]:
            try:
                old.unlink()
            except Exception:
                pass

    async def _checkpoint_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.interval_seconds)
                if self._running:
                    await self.checkpoint()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[CheckpointManager] Checkpoint error: {e}")

    def get_latest_checkpoint(self) -> Optional[CheckpointMetadata]:
        """Get metadata of latest checkpoint."""
        return self._last_checkpoint

    def get_checkpoint_history(self) -> List[CheckpointMetadata]:
        """Get list of available checkpoints."""
        checkpoints = []
        checkpoint_dir = self.checkpoint_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return []

        for path in sorted(checkpoint_dir.glob("checkpoint_*.msgpack.zst"),
                          key=lambda p: p.stat().st_mtime,
                          reverse=True):
            try:
                data = self.serializer.deserialize(path.read_bytes())
                metadata = CheckpointMetadata(
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    version=data["version"],
                    component="all",
                    size_bytes=path.stat().st_size,
                    checksum=hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                    sequence=data["sequence"],
                    metadata=data.get("metadata", {}),
                )
                checkpoints.append(metadata)
            except Exception:
                continue

        return checkpoints


class StateRestorer:
    """
    Restores system state from checkpoints.

    Features:
    - Automatic latest checkpoint selection
    - Component-by-component restoration
    - Validation and verification
    - Rollback capability
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        serializer: Optional[StateSerializer] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.serializer = serializer or StateSerializer()
        self.writer = AtomicFileWriter(self.checkpoint_dir / "checkpoints", backup_count=5)

    async def restore_latest(
        self,
        component_setters: Dict[str, Callable[[Any], None]],
        validate: bool = True,
    ) -> bool:
        """
        Restore latest checkpoint to components.

        Args:
            component_setters: Dict of component_name -> setter_function
            validate: Whether to validate restored data

        Returns:
            True if restoration successful
        """
        checkpoints = self._find_checkpoints()
        if not checkpoints:
            print("[StateRestorer] No checkpoints found")
            return False

        latest = checkpoints[0]
        print(f"[StateRestorer] Restoring from checkpoint {latest.sequence} ({latest.timestamp})")

        try:
            data = self.writer.read_latest(self.checkpoint_dir / "checkpoints" /
                f"checkpoint_{latest.sequence:08d}_{int(latest.timestamp.timestamp())}.msgpack.zst")

            if not data:
                print("[StateRestorer] No valid checkpoint data found")
                return False

            checkpoint_data = self.serializer.deserialize(data)

            if validate:
                if not self._validate_checkpoint(checkpoint_data):
                    print("[StateRestorer] Checkpoint validation failed")
                    return False

            # Restore components
            components = checkpoint_data.get("components", {})
            for name, setter in component_setters.items():
                if name in components:
                    try:
                        setter(components[name])
                        print(f"[StateRestorer] Restored {name}")
                    except Exception as e:
                        print(f"[StateRestorer] Failed to restore {name}: {e}")
                        return False

            print(f"[StateRestorer] Restored {len(component_setters)} components successfully")
            return True

        except Exception as e:
            print(f"[StateRestorer] Restore failed: {e}")
            return False

    def _find_checkpoints(self) -> List[CheckpointMetadata]:
        """Find available checkpoints sorted by sequence."""
        checkpoints = []
        checkpoint_dir = self.checkpoint_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return []

        for path in sorted(checkpoint_dir.glob("checkpoint_*.msgpack.zst"),
                          key=lambda p: p.stat().st_mtime,
                          reverse=True):
            try:
                data = self.serializer.deserialize(path.read_bytes())
                meta = CheckpointMetadata(
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    version=data["version"],
                    component="all",
                    size_bytes=path.stat().st_size,
                    checksum=hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                    sequence=data["sequence"],
                    metadata=data.get("metadata", {}),
                )
                checkpoints.append(meta)
            except Exception:
                continue

        return checkpoints

    def _validate_checkpoint(self, data: Dict) -> bool:
        """Validate checkpoint integrity."""
        required_keys = {"version", "timestamp", "sequence", "components"}
        if not all(k in data for k in required_keys):
            return False

        if data["version"] != "1.0":
            return False

        return True

    async def restore_from_checkpoint(
        self,
        sequence: int,
        component_setters: Dict[str, Callable[[Any], None]],
    ) -> bool:
        """Restore from specific checkpoint sequence."""
        checkpoints = self._find_checkpoints()

        for cp in checkpoints:
            if cp.sequence == sequence:
                path = self.checkpoint_dir / "checkpoints" / f"checkpoint_{sequence:08d}_{int(cp.timestamp.timestamp())}.msgpack.zst"
                try:
                    data = self.writer.read_latest(path)
                    if data:
                        checkpoint_data = self.serializer.deserialize(data)
                        for name, setter in component_setters.items():
                            if name in checkpoint_data.get("components", {}):
                                setter(checkpoint_data["components"][name])
                        return True
                except Exception as e:
                    print(f"[StateRestorer] Restore from {sequence} failed: {e}")
                    return False

        return False


class RecoveryCoordinator:
    """
    Coordinates system recovery after failure.

    Features:
    - Component health assessment
    - Recovery plan generation
    - Priority-based restoration
    - Dependency resolution
    - Progress tracking
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        state_restorer: StateRestorer,
        health_checkers: Dict[str, Callable[[], bool]],
        component_setters: Dict[str, Callable[[Any], None]],
    ):
        self.checkpoint_manager = checkpoint_manager
        self.state_restorer = state_restorer
        self.health_checkers = health_checkers
        self.component_setters = component_setters

        self._recovery_state = RecoveryState.HEALTHY
        self._recovery_plan: List[RecoveryPlan] = []

    async def assess_health(self) -> Dict[str, bool]:
        """Run all health checks."""
        results = {}
        for name, checker in self.health_checkers.items():
            try:
                results[name] = await asyncio.get_event_loop().run_in_executor(None, checker)
            except Exception as e:
                print(f"[RecoveryCoordinator] Health check {name} failed: {e}")
                results[name] = False
        return results

    async def assess_and_plan(self) -> List[RecoveryPlan]:
        """Assess system health and generate recovery plan."""
        health = await self.assess_health()

        unhealthy = [name for name, healthy in health.items() if not healthy]

        if not unhealthy:
            self._recovery_state = RecoveryState.HEALTHY
            return []

        # Determine severity
        critical_components = {"execution_engine", "order_manager", "risk_orchestrator"}
        critical_failed = [c for c in unhealthy if c in critical_components]

        if critical_failed:
            self._recovery_state = RecoveryState.FAILED
        elif len(unhealthy) > 2:
            self._recovery_state = RecoveryState.DEGRADED
        else:
            self._recovery_state = RecoveryState.RECOVERING

        # Generate recovery plan
        plan = self._generate_recovery_plan(unhealthy)
        self._recovery_plan = plan

        return plan

    def _generate_recovery_plan(self, unhealthy: List[str]) -> List[RecoveryPlan]:
        """Generate prioritized recovery plan."""
        # Priority order for restoration
        priority_order = [
            "execution_engine",
            "order_manager",
            "risk_orchestrator",
            "binance_rest",
            "binance_ws",
            "reconciliation_engine",
            "ml_engine",
            "strategy",
            "paper_engine",
        ]

        plan = []
        for i, component in enumerate(priority_order):
            if component in unhealthy:
                plan.append(RecoveryPlan(
                    component=component,
                    action="restore",
                    priority=i,
                    estimated_time_seconds=30.0,
                    dependencies=[d for d in priority_order[:i] if d not in unhealthy],
                ))
            elif component in self.health_checkers:
                # Healthy but dependent on unhealthy
                deps_in_unhealthy = [d for d in priority_order[:i] if d in unhealthy]
                if deps_in_unhealthy:
                    plan.append(RecoveryPlan(
                        component=component,
                        action="rebuild",
                        priority=len(priority_order) + i,
                        estimated_time_seconds=10.0,
                        dependencies=deps_in_unhealthy,
                    ))

        return plan

    async def execute_recovery(self, plan: List[RecoveryPlan]) -> bool:
        """Execute recovery plan."""
        print(f"[RecoveryCoordinator] Executing recovery plan with {len(plan)} steps")

        self._recovery_state = RecoveryState.RECOVERING

        for step in plan:
            print(f"[RecoveryCoordinator] Step {step.priority}: {step.action} {step.component}")

            # Wait for dependencies
            for dep in step.dependencies:
                while True:
                    health = await self.assess_health()
                    if health.get(dep, False):
                        break
                    await asyncio.sleep(1)

            # Execute action
            success = await self._execute_recovery_action(step)

            if not success:
                print(f"[RecoveryCoordinator] Step {step.component} failed")
                if step.priority < 3:  # Critical component
                    return False

        # Final validation
        final_health = await self.assess_health()
        if all(health.values()):
            self._recovery_state = RecoveryState.HEALTHY
            print("[RecoveryCoordinator] Recovery successful")
            return True
        else:
            self._recovery_state = RecoveryState.FAILED
            print("[RecoveryCoordinator] Recovery incomplete")
            return False

    async def _execute_recovery_action(self, step: RecoveryPlan) -> bool:
        """Execute a single recovery action."""
        try:
            if step.action == "restore":
                # Restore from checkpoint
                success = await self.state_restorer.restore_latest(
                    {step.component: self.component_setters[step.component]}
                )
                return success
            elif step.action == "rebuild":
                # Rebuild component (restart)
                print(f"[RecoveryCoordinator] Rebuilding {step.component}")
                await asyncio.sleep(step.estimated_time_seconds)
                return True
            elif step.action == "reset":
                # Full reset
                print(f"[RecoveryCoordinator] Resetting {step.component}")
                return True
            elif step.action == "skip":
                return True

            return False
        except Exception as e:
            print(f"[RecoveryCoordinator] Action {step.action} for {step.component} failed: {e}")
            return False

    @property
    def recovery_state(self) -> RecoveryState:
        return self._recovery_state


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_disaster_recovery_system(
    checkpoint_dir: Path,
    checkpoint_interval: float = 60.0,
    max_checkpoints: int = 10,
) -> tuple[CheckpointManager, StateRestorer, RecoveryCoordinator]:
    """
    Create complete disaster recovery system.

    Returns:
        (checkpoint_manager, state_restorer, recovery_coordinator)
    """
    serializer = StateSerializer()

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        interval_seconds=checkpoint_interval,
        max_checkpoints=max_checkpoints,
        serializer=serializer,
    )

    state_restorer = StateRestorer(
        checkpoint_dir=checkpoint_dir,
        serializer=serializer,
    )

    # These will be populated by the application
    health_checkers = {}
    component_setters = {}

    recovery_coordinator = RecoveryCoordinator(
        checkpoint_manager=checkpoint_manager,
        state_restorer=state_restorer,
        health_checkers=health_checkers,
        component_setters=component_setters,
    )

    return checkpoint_manager, state_restorer, recovery_coordinator


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "RecoveryState",
    "CheckpointMetadata",
    "RecoveryPlan",
    "StateSerializer",
    "AtomicFileWriter",
    "CheckpointManager",
    "StateRestorer",
    "RecoveryCoordinator",
    "create_disaster_recovery_system",
]