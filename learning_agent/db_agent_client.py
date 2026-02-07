
import os
import httpx
from typing import List, Dict, Optional, Union
import logging
from .models import Trade

# --- Configuration ---
DB_AGENT_BASE_URL = os.getenv("DB_AGENT_URL")
DB_AGENT_API_KEY = os.getenv("DB_AGENT_API_KEY")

# --- API Client ---
async def fetch_trade_history(
    account_id: Union[int, str],
    asset_id: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> List[Trade]:
    """
    Fetches trade history from the Database Agent.

    Args:
        account_id: The ID of the account to fetch history for.
        asset_id: If provided, fetches trades only for a specific asset.
        correlation_id: Optional correlation ID for tracing.

    Returns:
        A list of Trade objects. Returns an empty list if the fetch fails.
    """
    if not DB_AGENT_BASE_URL:
        logging.error("DB_AGENT_URL environment variable is not set. Cannot fetch trade history.")
        return []

    endpoint = f"{DB_AGENT_BASE_URL}/accounts/{account_id}/trade_history"
    params = {}
    if asset_id:
        params["asset_id"] = asset_id

    headers = {}
    if DB_AGENT_API_KEY:
        headers["X-API-KEY"] = DB_AGENT_API_KEY
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params, headers=headers, timeout=10.0)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes

            response_json = response.json()

            # The Database Agent now returns a StandardAgentResponse
            if isinstance(response_json, dict) and "data" in response_json:
                trade_data = response_json["data"]
            else:
                trade_data = response_json

            # Map Database Agent fields to Learning Agent Trade model
            processed_trades = []
            for data in trade_data:
                # Map 'symbol' to 'asset_id' if 'asset_id' is missing (Database Agent uses 'symbol')
                if not data.get("asset_id") and data.get("symbol"):
                    data["asset_id"] = data["symbol"]

                try:
                    processed_trades.append(Trade(**data))
                except Exception as e:
                    logging.warning(f"Skipping trade {data.get('trade_id')} due to parsing error: {e}")

            logging.info(f"Successfully fetched {len(processed_trades)} trades for asset '{asset_id}' from the Database Agent.")
            return processed_trades

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
