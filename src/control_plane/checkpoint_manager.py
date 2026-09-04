"""
QuantAI Checkpoint Manager
Manages checkpoints for disaster recovery and state persistence
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .supervisor import SupervisorState


@dataclass
class Checkpoint:
    """Checkpoint for disaster recovery"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    iteration: int = 0
    stage: str = "research"
    state: Any = None
    evidence: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "iteration": self.iteration,
            "stage": self.stage,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            iteration=data["iteration"],
            stage=data["stage"],
            metadata=data.get("metadata", {})
        )


class CheckpointManager:
    """
    Manages checkpoints for disaster recovery and state persistence.
    Saves periodic snapshots of the entire system state.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "data/checkpoints",
        interval_seconds: float = 60.0,
        max_checkpoints: int = 10
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_seconds
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Any] = []
        self._save_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start periodic checkpointing"""
        if self._running:
            return
        self._running = True
        self._save_task = asyncio.create_task(self._save_loop())
    
    async def stop(self) -> None:
        """Stop checkpointing"""
        self._running = False
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
    
    async def _save_loop(self) -> None:
        """Background task to save checkpoints periodically"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                # Check if it's time for a checkpoint
                # Actual saving is triggered by supervisor
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    async def save(self, checkpoint: Any) -> str:
        """Save a checkpoint"""
        # Create checkpoint directory
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        checkpoint_file = f"checkpoint_{checkpoint.id}_{timestamp}.pkl"
        file_path = self.checkpoint_dir / checkpoint_file
        
        try:
            # Save using pickle for complex objects
            with open(file_path, 'wb') as f:
                import pickle
                pickle.dump(checkpoint, f)
            
            # Also save metadata as JSON
            meta_file = self.checkpoint_dir / f"{checkpoint_file}.meta.json"
            with open(meta_file, 'w') as f:
                json.dump(checkpoint.to_dict() if hasattr(checkpoint, 'to_dict') else {}, f, default=str)
            
            # Clean old checkpoints
            await self._cleanup_old()
            
            return str(checkpoint_file)
        
        except Exception as e:
            print(f"[CheckpointManager] Error saving checkpoint: {e}")
            raise
    
    async def load_latest(self) -> Optional[Any]:
        """Load the most recent checkpoint"""
        files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if not files:
            return None
        
        # Sort by modification time, newest first
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        try:
            with open(files[0], 'rb') as f:
                import pickle
                return pickle.load(f)
        except Exception:
            return None
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Any]:
        """Load specific checkpoint by ID"""
        files = list(self.checkpoint_dir.glob(f"checkpoint_{checkpoint_id}_*.pkl"))
        if not files:
            return None
        
        try:
            with open(files[0], 'rb') as f:
                import pickle
                return pickle.load(f)
        except Exception:
            return None
    
    async def _cleanup_old(self) -> None:
        """Remove old checkpoints beyond max_checkpoints"""
        files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        for old_file in files[self.max_checkpoints:]:
            try:
                old_file.unlink()
                # Also remove meta file
                meta_file = old_file.with_suffix('.meta.json')
                if meta_file.exists():
                    meta_file.unlink()
            except Exception:
                pass
    
    async def emergency_save(self, checkpoint: Any) -> str:
        """Emergency save - saves immediately without waiting"""
        return await self.save(checkpoint)
    
    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        for meta_file in self.checkpoint_dir.glob("*.meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    checkpoints.append(meta)
            except Exception:
                pass
        return sorted(checkpoints, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    async def stop(self) -> None:
        """Stop checkpoint manager"""
        self._running = False


__all__ = [
    "Checkpoint",
    "CheckpointManager",
]