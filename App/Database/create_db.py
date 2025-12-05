"""
Utility script to initialise the database schema.

Running this script creates all tables defined in ``db_models.py`` in the
database configured in ``database.py``.  It should be executed once after
you define or modify your ORM models.  If tables already exist, this script
has no effect.

Usage:
    python create_db.py
"""

from App.Database.database import engine, Base

# Import ORM models so that SQLAlchemy registers them with the metadata.  If
# you add new model classes to db_models.py, be sure to import them here.
from Models import db_models  # noqa: F401  # pylint: disable=unused-import


def init_db() -> None:
    """Create all tables in the database based on the ORM models."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    init_db()
