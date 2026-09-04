import sys
sys.path.insert(0, '.')

import asyncio
import traceback
from src.control_plane.quant_researcher import QuantResearcher

async def test():
    researcher = QuantResearcher()
    try:
        result = await researcher.run_tournament()
        print(f'Result type: {type(result)}')
        if isinstance(result, dict):
            print(f'Result keys: {list(result.keys())}')
            if 'tournament' in result:
                t = result['tournament']
                print(f'Tournament type: {type(t)}')
                if isinstance(t, dict):
                    print(f'Tournament keys: {list(t.keys())}')
                    for k, v in t.items():
                        print(f'  {k}: {type(v)}')
            else:
                print(f'Tournament is not dict: {type(t)}')
        else:
            print(f'Result is not dict: {type(result)}')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

import asyncio
asyncio.run(test())