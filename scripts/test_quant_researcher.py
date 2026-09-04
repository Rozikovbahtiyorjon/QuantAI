import sys
sys.path.insert(0, '.')

import asyncio
from src.control_plane.quant_researcher import QuantResearcher

async def test():
    researcher = QuantResearcher()
    result = await researcher.run_tournament()
    print(f'Recommendation: {result["recommendation"]}')
    tournament = result.get("tournament", {})
    if "best" in tournament:
        print(f'Best: {tournament["best"]}')
    else:
        print(f'Tournament keys: {list(tournament.keys())}')

asyncio.run(test())