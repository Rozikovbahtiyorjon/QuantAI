import sys
sys.path.insert(0, '.')
import asyncio
from src.control_plane.supervisor import AISupervisor, SupervisorConfig
from src.control_plane.task_manager import Task, TaskPriority

async def test():
    cfg = SupervisorConfig(max_concurrent_tasks=2, retry_delay_seconds=1)
    sup = AISupervisor(cfg)
    print('supervisor import OK')
    print('agents:', list(sup.agent_router.agents.keys()))
    
    # Test routing for research stage with alpha_research capability
    t = Task(name='Tournament 4h B', description='Run breakout family WF', stage='research', priority=TaskPriority.HIGH, metadata={'required_capabilities':['alpha_research'], 'family':'B', 'timeframe':'4h'})
    agent = await sup.agent_router.route(task=t, stage='research', state=sup.state)
    print(f'routed agent for research+alpha_research: {agent}')
    
    # Test routing for wfo
    t2 = Task(name='WFO 4h Breakout', description='WFO PF median>1.05', stage='wfo', priority=TaskPriority.HIGH, metadata={'family':'B'})
    agent2 = await sup.agent_router.route(task=t2, stage='wfo', state=sup.state)
    print(f'routed agent for wfo: {agent2}')
    
    # Direct quant_researcher execution via agent_router
    print('testing _run_quant_researcher via agent_router.execute...')
    result = await sup.agent_router.execute(agent_type='quant_researcher', task=t, state=sup.state)
    print(f"execute success: {result.get('success')}")
    if result.get('success'):
        inner = result.get('result', {})
        print(f"inner success: {inner.get('success')}")
        tour = inner.get('tournament', {})
        print(f"tournament keys: {list(tour.keys()) if isinstance(tour, dict) else type(tour)}")
        if 'A' in tour:
            print(f"A PF {tour['A'].get('backtest_pf')}")
        print(f"best: {inner.get('best')}")
        print(f"research_report: {inner.get('research_report')}")

asyncio.run(test())
