import sys
sys.path.insert(0, '.')
import asyncio
from src.control_plane.supervisor import create_supervisor
from src.control_plane import SupervisorConfig
from src.control_plane.task_manager import Task, TaskPriority

async def main():
    cfg = SupervisorConfig(max_concurrent_tasks=1, retry_delay_seconds=2)
    sup = await create_supervisor(cfg)
    # Submit research task that will trigger real tournament
    t = Task(name="Tournament 4h Research", description="Run 3-family tournament on 4h", stage="research", priority=TaskPriority.HIGH, metadata={"required_capabilities": ["alpha_research"]})
    await sup.submit_task(t)
    print(f"Submitted {t.name} stage={t.stage}")
    # Let supervisor run 2 cycles (~4 min)
    await asyncio.sleep(180)
    print(await sup.get_status())
    # Show evidence
    ev = await sup.evidence_manager.get_recent(limit=5)
    for e in ev:
        print(e.type, str(e.data)[:300])
    await sup.stop()
    print("Stopped")

asyncio.run(main())
