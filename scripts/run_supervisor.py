import sys
sys.path.insert(0, '.')
import asyncio
from src.control_plane.supervisor import create_supervisor
from src.control_plane import SupervisorConfig
from src.control_plane.task_manager import Task, TaskPriority

async def main():
    cfg = SupervisorConfig(max_concurrent_tasks=2, retry_delay_seconds=3)
    sup = await create_supervisor(cfg)
    # Submit sample tasks
    t1 = Task(name="Backtest 4h Baseline", description="Run backtest PF>1.05", stage="backtest", priority=TaskPriority.HIGH, metadata={"required_capabilities": ["alpha_research"]})
    t2 = Task(name="WFO 4h Breakout", description="WFO PF median>1.05", stage="wfo", priority=TaskPriority.HIGH)
    await sup.submit_task(t1)
    await sup.submit_task(t2)
    print(f"Supervisor started stage={sup.state.current_stage} tasks={[t1.name, t2.name]}")
    # Run for 30s
    await asyncio.sleep(30)
    print(await sup.get_status())
    await sup.stop()
    print("Supervisor stopped")

asyncio.run(main())
