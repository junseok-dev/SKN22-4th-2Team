import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

# Load .env
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

try:
    print("Testing PatentAgent initialization...")
    from src.patent_agent import PatentAgent
    agent = PatentAgent()
    print("PatentAgent initialized successfully!")
    
    # Test a small search
    import asyncio
    async def test_search():
        print("Testing search...")
        try:
            results = await agent.search_with_grading("test idea")
            print(f"Search successful! Found {len(results)} results.")
        except Exception as e:
            print(f"Search failed: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test_search())

except Exception as e:
    print(f"FAILED to initialize PatentAgent: {e}")
    import traceback
    traceback.print_exc()
