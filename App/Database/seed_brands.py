"""
Populate the ``brands`` table with a set of known beverage brands and
their categories.  Run this script once after your database schema has
been created to insert brand records into the database.  It checks for
existing entries to avoid duplicate inserts.

You can adjust the list of brands or categories as needed.  Categories
group brands into high‑level families (e.g., ``cola``, ``fruit soda``,
``other``) which are used by the recommendation engine to suggest
alternative drinks when a favourite brand is unavailable.
"""

from .database import SessionLocal
from Models.db_models import Brand


def seed_brands() -> None:
    """Insert known brands into the database if they don't already exist."""
    # Define the brands and their categories.  Adjust categories to suit
    # your classification scheme.  Brands not present in this list will
    # need to be added manually or through user input.
    brand_data = [
        {"name": "Coca-Cola", "category": "cola"},
        {"name": "Pepsi", "category": "cola"},
        {"name": "American Cola", "category": "cola"},
        {"name": "RC Cola", "category": "cola"},
        {"name": "Fanta", "category": "fruit soda"},
        {"name": "Mirinda", "category": "fruit soda"},
        {"name": "La Casera", "category": "fruit soda"},
        {"name": "Sprite", "category": "fruit soda"},
        {"name": "Teem", "category": "fruit soda"},
        {"name": "Bigi", "category": "other"},
        {"name": "Mountain Dew", "category": "other"},
        {"name": "7Up", "category": "other"},
        {"name": "Schweppes", "category": "other"},
        {"name": "Fayrouz", "category": "other"},
    ]

    session = SessionLocal()
    try:
        for brand in brand_data:
            existing_brand = (
                session.query(Brand).filter_by(name=brand["name"]).first()
            )
            if existing_brand:
                continue  # Skip brands that already exist
            session.add(Brand(name=brand["name"], category=brand["category"]))
        session.commit()
        print(f"Seeded {len(brand_data)} brands into the database (duplicates skipped).")
    finally:
        session.close()


if __name__ == "__main__":
    seed_brands()
