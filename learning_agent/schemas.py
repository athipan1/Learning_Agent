
from sqlalchemy import Column, String, Float, DateTime, Integer, BigInteger, Numeric
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Order(Base):
    """
    SQLAlchemy model for the 'orders' table, matching the Database Agent's schema.
    Used for direct database access to trade history.
    """
    __tablename__ = 'orders'

    order_id = Column(Integer, primary_key=True)
    account_id = Column(Integer, nullable=False)
    symbol = Column(String, nullable=False)
    order_type = Column(String, nullable=False)
    quantity = Column(BigInteger, nullable=False)
    price = Column(Numeric(18, 5))
    status = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class BiasState(Base):
    """
    SQLAlchemy model for persisting the BIAS_STATE of each asset.
    """
    __tablename__ = 'bias_states'

    asset_id = Column(String, primary_key=True, index=True)
    bull_bias = Column(Float, nullable=False, default=0.0)
    bear_bias = Column(Float, nullable=False, default=0.0)
    vol_bias = Column(Float, nullable=False, default=0.0)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "bull_bias": self.bull_bias,
            "bear_bias": self.bear_bias,
            "vol_bias": self.vol_bias
        }
