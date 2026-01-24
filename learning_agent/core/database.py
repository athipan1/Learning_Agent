
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .schemas import Base  # Import the Base from our new schemas

# --- Database Configuration ---
# Use a more robust way to get the database URL
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logging.warning("DATABASE_URL not found in environment. Using default SQLite for local dev.")
    DATABASE_URL = "sqlite:///./learning_agent.db"  # Default to a local SQLite DB

SQLALCHEMY_ECHO = os.getenv("SQLALCHEMY_ECHO", "False").lower() in ("true", "1", "t")

try:
    engine = create_engine(DATABASE_URL, echo=SQLALCHEMY_ECHO)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logging.info("Database engine and session created successfully.")
except Exception as e:
    logging.critical(f"Failed to create database engine: {e}")
    raise

def init_db():
    """
    Initializes the database by creating all tables defined in schemas.py.
    This is a critical step for the first run or for setting up a new DB.
    """
    try:
        logging.info("Initializing database and creating tables...")
        # Create all tables that inherit from Base
        Base.metadata.create_all(bind=engine)
        logging.info("Database tables created successfully.")
    except Exception as e:
        logging.error(f"Error initializing database tables: {e}")
        raise

def get_db_session():
    """
    Provides a SQLAlchemy database session.
    Should be used with a context manager (e.g., `with get_db_session() as db:`).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
