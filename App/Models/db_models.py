"""
ORM model definitions for the beverage outlet application.

This module defines SQLAlchemy ORM classes representing users, outlets,
brands and the many‑to‑many relationships between them.  These models
provide the foundation for persisting and querying data in the PostgreSQL
database configured in ``app/database/database.py``.

Each class inherits from ``Base`` imported from ``database``.  When you run
``create_db.py``, SQLAlchemy will create the corresponding tables in your
Neon database based on these definitions.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Boolean,
    LargeBinary,
)
from sqlalchemy.orm import relationship
from datetime import datetime

from Database.database import Base


class User(Base):
    """Represents a user of the application.

    Users can be consumers or outlet owners.  They have a preferred
    beverage (``favourite_drink``) and optional home coordinates for
    location‑based features.
    """

    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    role = Column(String, nullable=False)  # "consumer" or "owner"
    sub_role = Column(String, nullable=True)  # e.g. "hawker", "Kiosk", "shop"
    display_name = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    display_pic = Column(LargeBinary, nullable=True)
    favourite_drink = Column(String, nullable=True)
    home_latitude = Column(Float, nullable=True)
    home_longitude = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to               s owned by the user.  A consumer will have an
    # empty list here, but an owner can have multiple outlets.
    outlets = relationship("Outlet", back_populates="owner")

    # Relationship to user ratings of outlets
    outlet_ratings = relationship("OutletRating", back_populates="user")


class Outlet(Base):
    """Represents a physical outlet (shop, kiosk, hawker, etc.)."""

    __tablename__ = "outlets"

    outlet_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.user_id"))
    display_pic = Column(LargeBinary, nullable=True)
    outlet_type = Column(String, nullable=False)  # e.g. "kiosk", "mobile", "shop"
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    is_mapped = Column(Boolean, default=True)
    contact_info = Column(String, nullable=True)  # <-- add this line
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to the owning user
    owner = relationship("User", back_populates="outlets")
    # Relationship to brands stocked at this outlet
    brands = relationship("OutletBrand", back_populates="outlet")
    # Relationship to ratings left by users
    ratings = relationship("OutletRating", back_populates="outlet")


class Brand(Base):
    """Represents a beverage brand (e.g., Pepsi, Fanta)."""

    __tablename__ = "brands"

    brand_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    category = Column(String, nullable=False)  # e.g. "cola", "fruit soda", "other"

    # Relationship to outlets stocking this brand
    outlets = relationship("OutletBrand", back_populates="brand")


class OutletBrand(Base):
    """Associative table linking outlets and brands with stock status."""

    __tablename__ = "outlet_brands"

    outlet_id = Column(Integer, ForeignKey("outlets.outlet_id"), primary_key=True)
    brand_id = Column(Integer, ForeignKey("brands.brand_id"), primary_key=True)
    stock_status = Column(
        String,
        nullable=False,
    )  # e.g. "Well stocked", "Partially stocked", "Almost empty"
    last_updated = Column(DateTime, default=datetime.utcnow)

    # Relationships back to parent tables
    outlet = relationship("Outlet", back_populates="brands")
    brand = relationship("Brand", back_populates="outlets")


class UserInteraction(Base):
    """Logs user actions such as viewing, accepting or rejecting suggestions."""

    __tablename__ = "user_interactions"

    interaction_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    brand_id = Column(Integer, ForeignKey("brands.brand_id"), nullable=True)
    outlet_id = Column(Integer, ForeignKey("outlets.outlet_id"))
    action = Column(
        String,
        nullable=False,
    )  # e.g. "view", "suggested", "accepted", "rejected", "called"
    timestamp = Column(DateTime, default=datetime.utcnow)


class FavouriteOutlet(Base):
    """Stores a user's favourite outlets and an optional rating."""

    __tablename__ = "favourite_outlets"

    user_id = Column(Integer, ForeignKey("users.user_id"), primary_key=True)
    outlet_id = Column(Integer, ForeignKey("outlets.outlet_id"), primary_key=True)
    rating = Column(Integer, nullable=True)  # user rating of the outlet (1–5)
    last_visited = Column(DateTime, default=datetime.utcnow)


class OutletRating(Base):
    """Represents a rating and optional comment left by a user for an outlet."""

    __tablename__ = "outlet_ratings"

    rating_id = Column(Integer, primary_key=True, index=True)
    outlet_id = Column(Integer, ForeignKey("outlets.outlet_id"))
    user_id = Column(Integer, ForeignKey("users.user_id"))
    rating = Column(Integer, nullable=False)  # rating value, e.g. 1–5 stars
    comment = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships back to users and outlets
    outlet = relationship("Outlet", back_populates="ratings")
    user = relationship("User", back_populates="outlet_ratings")
