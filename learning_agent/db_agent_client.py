
import os
import httpx
from typing import List, Dict, Optional
import logging
from .models import Trade

# --- Configuration ---
DB_AGENT_BASE_URL = os.getenv("DB_AGENT_URL", "http://localhost:8001/api/v1")
if not DB_AGENT_BASE_URL:
    raise ValueError("DB_AGENT_URL environment variable is not set.")

# --- API Client ---
async def fetch_trade_history(asset_id: Optional[str] = None) -> List[Trade]:
    """
    Fetches trade history from the Database Agent.

    Args:
        asset_id: If provided, fetches trades only for a specific asset.

    Returns:
        A list of Trade objects. Returns an empty list if the fetch fails.
    """
    endpoint = f"{DB_AGENT_BASE_URL}/trades"
    params = {}
    if asset_id:
        params["asset_id"] = asset_id

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params, timeout=10.0)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes

            trade_data = response.json()

            # Parse the raw dictionary data into Pydantic Trade models
            trades = [Trade(**data) for data in trade_data]

            logging.info(f"Successfully fetched {len(trades)} trades for asset '{asset_id}' from the Database Agent.")
            return trades

    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error occurred while fetching trades for asset '{asset_id}': {e.response.status_code} - {e.response.text}")
        return []
    except httpx.RequestError as e:
        logging.error(f"An error occurred while requesting trades for asset '{asset_id}' from the Database Agent: {e}")
        return []
    except Exception as e:
        # Catch any other unexpected errors, including JSON parsing errors
        logging.error(f"An unexpected error occurred while fetching trade history for asset '{asset_id}': {e}")
        return []
