"""
Database configuration module.

This module defines the SQLAlchemy engine, session factory, and base class for
declaring ORM models. It uses a PostgreSQL connection string hosted on Neon.

Replace the `DATABASE_URL` value with the appropriate connection string for
your environment. When deploying to production (e.g., on Streamlit Cloud),
consider reading this value from environment variables or Streamlit secrets
instead of hard‑coding it.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
import os
import streamlit as st
from dotenv import load_dotenv

# Neon connection string with SSL.  This should be stored securely in
# production.  If you deploy to Streamlit Cloud, put this value in
# `.streamlit/secrets.toml` and read it from `st.secrets` instead.
load_dotenv()
database_url = (
    st.secrets.get("database", {}).get("url")
    or os.environ.get("DATABASE_URL")
    or "sqlite:///./app.db"  # fallback if nothing else is provided
)


# Create the SQLAlchemy engine.  `pool_pre_ping=True` ensures that
# connections are checked before being reused, which helps avoid broken
# connections in long‑running applications.
engine = create_engine(
    database_url,
    pool_pre_ping=True,
)

# Create a configured "Session" class.  Each instance of SessionLocal
# will be a database session bound to the engine defined above.  Use
# `SessionLocal()` to create a session when interacting with the database.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all ORM models.  Each model class should inherit from
# this base class so that SQLAlchemy knows how to map it to a table.
Base = declarative_base()


#Update the tables to include display pic and name 
with engine.connect() as conn:
    # Add new columns if they don’t exist
    conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR(255);"))
    conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS display_pic BYTEA;"))
    conn.execute(text("ALTER TABLE outlets ADD COLUMN IF NOT EXISTS display_pic BYTEA;"))
    conn.execute(text("ALTER TABLE outlets ADD COLUMN IF NOT EXISTS contact_info VARCHAR(255);"))
    

    #Drop columns
    conn.execute(text("ALTER TABLE users DROP COLUMN IF EXISTS username;"))

    conn.commit()