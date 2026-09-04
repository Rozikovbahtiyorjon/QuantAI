import sys
sys.path.insert(0, '.')
import asyncio
from src.control_plane.supervisor import create_supervisor, SupervisorConfig
from src.control_plane.task_manager import Task, TaskPriority

async def test():
    cfg = SupervisorConfig(max_concurrent_tasks=2, retry_delay_seconds=2)
    sup = await create_supervisor(cfg)
    print(f"Supervisor started: stage={sup.state.current_stage}")
    
    # Submit tournament task for research stage
    t = Task(
        name="Tournament 4h - 3 families",
        description="Run 3-family tournament on BTCUSDT 4h with WF validation",
        stage="research",
        priority=TaskPriority.HIGH,
        metadata={
            "required_capabilities": ["alpha_research", "regime_detection"],
            "family": "B",
            "timeframe": "4h",
            "symbols": ["BTCUSDT"]
        }
    )
    task_id = await sup.submit_task(t)
    print(f"Submitted task: {t.name} (id={task_id[:8]})")
    
    # Run for 90 seconds to allow one full cycle
    print("Running supervisor cycle for 90s...")
    await asyncio.sleep(90)
    
    status = await sup.get_status()
    print(f"Status: {status}")
    print(f"Active tasks: {len(sup.state.active_tasks)}")
    print(f"Completed: {len(sup.state.completed_tasks)}")
    print(f"Failed: {len(sup.state.failed_tasks)}")
    print(f"Errors: {sup.state.errors[-3:] if sup.state.errors else 'none'}")
    
    # Check evidence
    evidence = await sup.evidence_manager.get_recent(limit=10)
    print(f"Evidence stored: {len(evidence)} items")
    for e in evidence[-3:]:
        print(f"  {e.type.value} from {e.source}: {list(e.data.keys()) if isinstance(e.data, dict) else type(e.data)}")
    
    await sup.stop()
    print("Supervisor stopped")

asyncio.run(test())