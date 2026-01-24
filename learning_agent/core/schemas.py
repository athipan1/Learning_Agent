
from sqlalchemy import (
    Column, String, Float, DateTime, Boolean, JSON, Integer, BigInteger,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone
import uuid

Base = declarative_base()


class Strategy(Base):
    """
    Configuration for a single trading strategy.
    The Learning Agent will query this table to find active strategies to run.
    """
    __tablename__ = 'strategies'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_name = Column(String, nullable=False, unique=True, index=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    config_json = Column(JSON, nullable=True)  # For strategy-specific params
    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class HistoricalMarketData(Base):
    """
    Stores historical OHLCV data for various assets.
    """
    __tablename__ = 'historical_market_data'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)

    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', name='_symbol_timestamp_uc'),
    )


class LearningResult(Base):
    """
    Stores the primary, actionable output of a learning pipeline run for a specific strategy.
    This is the table the live API will query.
    """
    __tablename__ = 'learning_results'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_id = Column(String, ForeignKey('strategies.id'), nullable=False, index=True)
    bias_adjustment = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)  # e.g., "LOW", "MEDIUM", "HIGH"
    market_regime = Column(String, nullable=False) # e.g., "TRENDING", "SIDEWAYS"
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class LearningLog(Base):
    """
    Stores detailed, secondary outputs (logs, metrics) from a pipeline run for audit and analysis.
    """
    __tablename__ = 'learning_logs'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String, index=True, nullable=False) # A unique ID for each pipeline execution
    strategy_id = Column(String, ForeignKey('strategies.id'), nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    log_data = Column(JSON, nullable=False) # Detailed metrics, raw features, etc.


# The old BiasState is being replaced by LearningResult, which is more structured.
# The concept of a simple "bias" is now part of the LearningResult's "bias_adjustment".
# If more complex state per strategy/asset is needed later, a new table can be designed.
# For now, we are removing the old BiasState table to align with the new architecture.
