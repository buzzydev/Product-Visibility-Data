"""
Streamlit application for the Product visibility project.

This app provides two user roles: consumers and outlet owners.  Consumers
can sign up with a favourite drink and their location, log in, and see
nearby outlets along with predicted stock conditions and alternative
recommendations.  Outlet owners can register their business, specify
what brands they stock, and view basic analytics.  The app also lays
the groundwork for future features such as ordering and real‑time
hawker alerts.

Note: This implementation emphasises readability and demonstrates
basic interactions with the database and machine‑learning models.  It
does not cover every nuance of the planned recommendation logic.  You
can extend these foundations by logging user interactions and
progressively training more personalised models.
"""

import math
import io
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from passlib.context import CryptContext
# Import IntegrityError so we can handle duplicate inserts gracefully
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func, desc  # for aggregation queries
from sqlalchemy.orm import Session
from PIL import Image  # for resizing profile pictures

# We will lazily import joblib only when needed to avoid unnecessary
# overhead during app startup.  joblib is used to load the machine‑learning
# models saved during the training phase.
import joblib

from Database.database import SessionLocal
from Models.db_models import (
    User,
    Outlet,
    Brand,
    OutletBrand,
    FavouriteOutlet,
    UserInteraction,
    OutletRating,
)


# ---------------------------------------------------------------------------
# Model loading utilities
#
# To demonstrate how the trained machine‑learning models can be used in the
# application, we include helper functions that load the stock condition
# classifier and perform a simple prediction based on a few outlet features.
# The stock condition model was trained using the cleaned dataset with
# features such as one‑hot encodings of the outlet type and counts of
# packaged products and display methods.  In this MVP we only have
# the number of brands stocked and the outlet type available, so the
# prediction may be approximate.  You can extend the feature set and
# retrain the model to improve accuracy as you capture more information.


@st.cache_resource(show_spinner=False)
def load_stock_condition_model() -> Optional[object]:
    """Load the stock condition classifier from disk.

    Returns None if the model file cannot be found.  We use a cache to
    avoid reloading the model on every interaction.
    """
    model_path = Path("models/stock_condition_model.pkl")
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            # If loading fails, return None and log the error in Streamlit.
            st.warning("Unable to load the stock condition model. Predictions will be unavailable.")
            return None
    else:
        # Model file is optional – the app still runs without ML integration.
        return None


@st.cache_resource(show_spinner=False)
def load_brand_presence_models() -> dict:
    """Load binary classifiers that predict brand availability.

    These models are stored in the `models/brand_presence_models` directory
    with filenames like ``Coca_Cola_presence_model.pkl``.  Each model is a
    scikit‑learn pipeline that includes preprocessing, so you can pass a
    simple DataFrame with the columns ``Type_Of_Outlet``, ``num_brands_present``,
    ``num_package_types`` and ``num_display_methods``.  The function
    returns a dictionary mapping brand names to their respective models.
    """
    models_dir = Path("models/brand_presence_models")
    presence_models: dict[str, object] = {}
    if models_dir.exists():
        for model_file in models_dir.glob("*_presence_model.pkl"):
            brand_key = model_file.stem.replace("_presence_model", "")
            try:
                presence_models[brand_key] = joblib.load(model_file)
            except Exception:
                st.warning(f"Failed to load model {model_file.name}")
    return presence_models

# ---------------------------------------------------------------------------
# Location mapping utilities
#
# We create a static mapping from high‑level areas to neighbourhoods (bus stops)
# with approximate latitude/longitude coordinates.  These coordinates are
# approximate and serve only to estimate distances to the outlets in our
# database.  In a production system you would use a geocoding API to convert
# user addresses to coordinates.

AREA_MAPPING: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Alimosho": {
        "Egbeda": (6.6184, 3.3214),
        "Akowonjo": (6.5865, 3.2981),
        "Ikotun": (6.5734, 3.2939),
        "Ipaja": (6.6300, 3.3080),
        "Ayobo": (6.6425, 3.2184),
    },
    "Ikeja": {
        "Allen Avenue": (6.5965, 3.3485),
        "Opebi": (6.6038, 3.3517),
        "Ikeja GRA": (6.5811, 3.3552),
        "Computer Village": (6.5937, 3.3430),
        "Maryland": (6.5704, 3.3602),
    },
    "Agege": {
        "Agege Market": (6.6205, 3.3093),
        "Dopemu": (6.6167, 3.3417),
        "Iyana Ipaja": (6.6342, 3.2653),
        "Oko Oba": (6.6568, 3.3196),
        "Abule Egba": (6.6418, 3.3255),
    },
    "Ajegunle": {
        "Boundary": (6.4527, 3.3676),
        "Wilmer": (6.4515, 3.3727),
        "Apapa Wharf": (6.4356, 3.3443),
        "Tincan": (6.4397, 3.3550),
        "Ajegunle Market": (6.4541, 3.3705),
    },
}
# Coordinate used when a user is outside Lagos; this forces the map view to show
# all outlets.
OUTSIDE_LAGOS_COORD: Tuple[float, float] = (5.0, 3.5)

# ---------------------------------------------------------------------------
# Helper to normalise user‑entered outlet types
# In the training data the model expects categories such as ``shop``,
# ``kiosk``, ``hawker`` and ``mobile``.  In the user interface we
# present more descriptive labels like "mobile (hawking)".  This helper
# collapses variants down to the core categories used during training.
def normalise_outlet_type(raw_type: str) -> str:
    """Normalise a raw outlet type string to one of the model categories.

    Parameters
    ----------
    raw_type : str
        The outlet type selected by the user (e.g. ``"shop"``,
        ``"kiosk (hawking)"`` or ``"mobile (hawking)"``).

    Returns
    -------
    str
        A simplified outlet type (``"shop"``, ``"kiosk"``, ``"hawker"`` or ``"mobile"``).
    """
    if not raw_type:
        return "shop"
    t = raw_type.lower()
    # Treat anything containing "mobile" or "hawking" as a hawker/mobile category
    if "mobile" in t or "hawking" in t:
        # The business logic here maps mobile hawkers to the ``hawker`` category
        # so that the model can handle them as informal vendors.
        return "hawker"
    if "kiosk" in t:
        return "kiosk"
    # Default to the first word (shop, supermarket etc.)
    return t.split()[0]


def predict_stock_condition(model: object, outlet_type: str, num_brands: int) -> Optional[str]:
    """Predict the stock condition for an outlet using the loaded model.

    Parameters
    ----------
    model : object
        A scikit‑learn compatible classifier loaded via joblib.  If None, the
        function returns None.
    outlet_type : str
        The type of outlet (e.g. 'shop', 'kiosk', 'hawker', 'mobile').  It
        should match one of the categories seen during training.
    num_brands : int
        The number of different brands stocked at this outlet.

    Returns
    -------
    Optional[str]
        The predicted stock condition label, or None if the model is not
        available or the input cannot be processed.

    Notes
    -----
    This helper constructs a simple feature vector consisting of one‑hot
    encoded outlet types and the number of brands.  It assumes zero values
    for the number of package and display types, since those details are
    not captured in the current database schema.  You should extend this
    function once you record additional features during sign‑up or outlet
    updates.
    """
    if model is None:
        return None
    # Define the list of outlet types in the order used during training.
    outlet_categories = ["shop", "kiosk", "hawker", "mobile"]
    # One‑hot encode the outlet type.
    type_vector = [1 if outlet_type == cat else 0 for cat in outlet_categories]
    # Construct the feature vector: [type_onehots, num_brands, num_packages, num_displays]
    # We set num_packages and num_displays to zero as we don't capture them yet.
    feature_vector = type_vector + [num_brands, 0, 0]
    # Use the classifier to predict the class label.  We wrap in a try/except
    # to handle unexpected input shapes.
    try:
        pred = model.predict([feature_vector])
        # `pred` is an array with a single element
        return str(pred[0])
    except Exception:
        return None

# Password hashing context.  We use bcrypt via passlib.  When
# deploying, you might enforce stronger policies or use an external
# authentication service.
# Use PBKDF2 with SHA‑256 to avoid dependency on the bcrypt backend
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash a plaintext password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a hashed one."""
    return pwd_context.verify(plain_password, hashed_password)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great‑circle distance between two points on the Earth.

    Returns distance in kilometres.
    """
    R = 6371  # Earth radius in kilometres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@st.cache_data(hash_funcs = {Session: lambda _: None}, show_spinner=False)
def load_brand_list(session: Session) -> List[str]:
    """Return a list of brand names from the database."""
    brands = session.query(Brand).all()
    return [b.name for b in brands]


def get_outlet_dataframe(session: Session) -> pd.DataFrame:
    """Retrieve all outlets with their coordinates and associated brands."""
    outlets = session.query(Outlet).all()
    data = []
    for o in outlets:
        brands = [ob.brand.name for ob in o.brands]
        data.append(
            {
                "outlet_id": o.outlet_id,
                "name": o.name,
                "type": o.outlet_type,
                "latitude": o.latitude,
                "longitude": o.longitude,
                "brands": ", ".join(brands),
            }
        )
    return pd.DataFrame(data)


def nearest_outlets(
    user_lat: float, user_lon: float, outlets_df: pd.DataFrame, k: int = 5
) -> pd.DataFrame:
    """Return the k nearest outlets to the given coordinates."""
    if outlets_df.empty:
        return pd.DataFrame()
    distances = outlets_df.apply(
        lambda row: haversine(user_lat, user_lon, row["latitude"], row["longitude"]),
        axis=1,
    )
    outlets_df = outlets_df.copy()
    outlets_df["distance_km"] = distances
    return outlets_df.sort_values("distance_km").head(k)


def register_consumer(session: Session) -> None:
    """Handle consumer sign‑up.

    This version asks the user for a display name, email, password,
    favourite drink, and an approximate location based on popular areas
    and neighbourhoods in Lagos.  Users may optionally upload a
    profile picture.  We use the ``AREA_MAPPING`` to derive a pair
    of latitude/longitude coordinates for the chosen area.  If the
    ``display_pic`` column exists on the ``User`` model, the image
    bytes are stored there; otherwise the picture is ignored.
    """
    st.subheader("Consumer Sign Up")
    # Basic inputs
    display_name = st.text_input("Display Name (username)")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    favourite = st.selectbox("Favourite Drink", load_brand_list(session), index=0)

    # Profile picture uploader.  We limit the file size and optionally
    # resize the image to avoid storing large files in the database.
    uploaded_file = st.file_uploader(
        "Upload a profile picture (optional, max 1 MB)",
        type=["jpg", "jpeg", "png"],
        key="consumer_profile_pic",
    )
    picture_bytes: Optional[bytes] = None
    if uploaded_file:
        if uploaded_file.size > 1_000_000:
            st.warning("The uploaded file is larger than 1 MB. Please choose a smaller image.")
        else:
            try:
                uploaded_file.seek(0)
                data = uploaded_file.getvalue()
                img = Image.open(io.BytesIO(data))
                img.thumbnail((300, 300))
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                picture_bytes = buf.getvalue()
            except Exception:
                st.warning("There was an error processing your image. It will be ignored.")
                picture_bytes = None

    # Ask for area and neighbourhood.  We provide a two‑stage drop‑down
    area_options = list(AREA_MAPPING.keys()) + ["Outside Lagos"]
    selected_area = st.selectbox("Where do you live?", area_options, key="consumer_area")
    if selected_area == "Outside Lagos":
        user_lat, user_lon = OUTSIDE_LAGOS_COORD
    else:
        neighbourhoods = list(AREA_MAPPING[selected_area].keys())
        selected_hood = st.selectbox(
            f"Select a neighbourhood in {selected_area}",
            neighbourhoods,
            key="consumer_neighbourhood",
        )
        user_lat, user_lon = AREA_MAPPING[selected_area][selected_hood]

    # Use a session flag to track whether the consumer account has been created.  This
    # avoids showing the Cancel button after a successful registration.
    if "consumer_created" not in st.session_state:
        st.session_state["consumer_created"] = False

    # Handle sign‑up submission
    if st.button("Sign Up", key="consumer_signup_button") and not st.session_state["consumer_created"]:
        # Validate passwords
        if password != confirm_password:
            st.warning("Passwords do not match.")
            return
        # Check if display name already exists
        if session.query(User).filter_by(display_name=display_name).first():
            st.warning(f"The display name '{display_name}' is already taken. Please choose another one.")
            return
        # Check if email already exists (avoid duplicate emails)
        if session.query(User).filter_by(email=email).first():
            st.warning("That email is already registered. Please choose another or log in.")
            return
        # Create user object
        user_kwargs = {
            "role": "consumer",
            "sub_role": None,
            "email": email,
            "password_hash": hash_password(password),
            "favourite_drink": favourite,
            "home_latitude": user_lat,
            "home_longitude": user_lon,
            "display_name": display_name,
        }
        user = User(**user_kwargs)
        # Persist profile picture if available
        if picture_bytes is not None and hasattr(user, "display_pic"):
            user.display_pic = picture_bytes
        session.add(user)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            st.error("An error occurred while creating your account. Please try again with a different display name or email.")
            return
        st.success("Account created!")
        # Mark account as created to hide the Cancel button
        st.session_state["consumer_created"] = True
    # After successful creation, offer a button to proceed to login.  When clicked,
    # we update the page and immediately render the login form within the same run.
    if st.session_state.get("consumer_created"):
        if st.button("Proceed to Login", key="consumer_proceed_login"):
            st.session_state["page"] = "Login"
            # Reset the created flag for future sign‑ups
            st.session_state["consumer_created"] = False
            login_user(session)
            return
    else:
        # Provide a cancel/back button to return to the Home page without creating an account
        if st.button("Cancel", key="consumer_cancel_button"):
            st.session_state["page"] = "Home"
            # Reset the flag in case it was set in a previous run
            st.session_state["consumer_created"] = False

def register_owner(session: Session) -> None:
    """Handle outlet owner sign‑up.

    Owners provide both a display name (used for login) and a business name
    for their outlet.  A file uploader allows them to upload a logo
    (stored on the Outlet record if the ``display_pic`` column exists).
    Locations are selected via high‑level area and neighbourhood drop‑downs
    similar to the consumer flow.  We ensure that neither the display
    name nor the business name is already in use.  Owners select the
    brands they stock; each is recorded in the ``OutletBrand`` table.
    """
    st.subheader("Outlet Owner Sign Up")
    # Ask for the owner's personal name.  This display name identifies the CEO and is reused across outlets.
    display_name = st.text_input("CEO / Display Name", key="owner_display")
    email = st.text_input("Email", key="owner_email")
    password = st.text_input("Password", type="password", key="owner_password")
    confirm_password = st.text_input(
        "Confirm Password", type="password", key="owner_confirm"
    )
    business_name = st.text_input("Business Name (outlet name)")
    contact_info = st.text_input("Contact information (phone or email)")
    # Business logo uploader
    logo_file = st.file_uploader(
        "Upload a business logo (optional, max 1 MB)", type=["jpg", "jpeg", "png"],
        key="owner_logo",
    )
    logo_bytes: Optional[bytes] = None
    if logo_file is not None:
        if logo_file.size > 1_000_000:
            st.warning("The uploaded logo is larger than 1 MB. Please choose a smaller file.")
        else:
            try:
                img = Image.open(logo_file)
                img.thumbnail((300, 300))
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                logo_bytes = buf.getvalue()
            except Exception:
                st.warning("There was an error processing your logo. It will be ignored.")
                logo_bytes = None
    # Outlet type selection.  We normalise the raw type to match the
    # categories used in the ML model via ``normalise_outlet_type``.  Note:
    # the displayed options use more descriptive labels but are mapped to
    # the simplified categories.
    raw_outlet_type = st.selectbox(
        "Outlet Type",
        [
            "shop",
            "kiosk (hawking)",
            "mobile (hawking)",
        ],
        key="owner_outlet_type",
    )
    outlet_type = normalise_outlet_type(raw_outlet_type)
    # Area and neighbourhood selection for the outlet location.
    area_options = list(AREA_MAPPING.keys()) + ["Outside Lagos"]
    selected_area = st.selectbox(
        "Select the general area of your outlet",
        area_options,
        key="owner_area",
    )
    if selected_area == "Outside Lagos":
        owner_lat, owner_lon = OUTSIDE_LAGOS_COORD
    else:
        hoods = list(AREA_MAPPING[selected_area].keys())
        selected_hood = st.selectbox(
            f"Select a neighbourhood in {selected_area}",
            hoods,
            key="owner_neighbourhood",
        )
        owner_lat, owner_lon = AREA_MAPPING[selected_area][selected_hood]
    # Brand selection (multi‑select)
    brand_options = load_brand_list(session)
    selected_brands = st.multiselect(
        "Select brands you stock", brand_options, default=[], key="owner_brands"
    )
    # Use a session flag to track whether the owner sign‑up has completed.  This avoids
    # showing the cancel button after a successful registration.
    if "owner_created" not in st.session_state:
        st.session_state["owner_created"] = False

    # Submit button
    if st.button("Register Outlet", key="owner_register_button") and not st.session_state["owner_created"]:
        # Validate passwords
        if password != confirm_password:
            st.warning("Passwords do not match.")
            return
        # Check if there is an existing owner with this email
        existing_user = session.query(User).filter_by(email=email).first()
        if existing_user:
            # Use the existing owner account and update display name if it has changed
            owner_user = existing_user
            if display_name and display_name != owner_user.display_name:
                owner_user.display_name = display_name
                session.commit()
        else:
            # Create a new owner account
            user_kwargs = {
                "role": "owner",
                "sub_role": outlet_type,
                "email": email,
                "password_hash": hash_password(password),
                "favourite_drink": None,
                "home_latitude": None,
                "home_longitude": None,
                "display_name": display_name,
            }
            owner_user = User(**user_kwargs)
            session.add(owner_user)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                st.error("An error occurred while creating the owner. Please try again with a different email.")
                return
        # Create a new outlet for this owner.  We allow owners to reuse business names across branches.
        outlet_kwargs = {
            "name": business_name,
            "owner_id": owner_user.user_id,
            "outlet_type": outlet_type,
            "latitude": owner_lat,
            "longitude": owner_lon,
            "is_mapped": True,
        }
        outlet = Outlet(**outlet_kwargs)
        # Attach logo if supported by schema
        if logo_bytes is not None and hasattr(outlet, "display_pic"):
            outlet.display_pic = logo_bytes
        session.add(outlet)
        # Associate selected brands with this outlet
        for b_name in selected_brands:
            brand = session.query(Brand).filter_by(name=b_name).first()
            if brand:
                session.add(
                    OutletBrand(
                        outlet_id=outlet.outlet_id,
                        brand_id=brand.brand_id,
                        stock_status="Well stocked",
                    )
                )
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            st.error("An error occurred while registering your outlet. Please try again.")
            return
        st.success("Outlet registered!")
        # Mark owner sign‑up complete to hide the cancel button
        st.session_state["owner_created"] = True
    # After registration, show proceed to login button
    if st.session_state.get("owner_created"):
        if st.button("Proceed to Login", key="owner_proceed_login"):
            st.session_state["page"] = "Login"
            st.session_state["owner_created"] = False
            login_user(session)
            return
    else:
        # Provide a cancel/back button to return to the Home page without creating an account
        if st.button("Cancel", key="owner_cancel_button"):
            st.session_state["page"] = "Home"
            st.session_state["owner_created"] = False

def login_user(session: Session) -> Optional[User]:
    """Simple login flow returning the authenticated user or None.

    This function accepts a display name (username) and looks up the
    appropriate field depending on the database schema.  It validates
    the password and returns the user object on success.
    """
    st.subheader("Login")
    display_name_input = st.text_input("Display Name", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    # Handle login submission
    if st.button("Log In", key="login_submit_button"):
        # Always search by display_name
        query_user = session.query(User).filter_by(display_name=display_name_input).first()
        # Verify credentials
        if query_user and verify_password(password, query_user.password_hash):
            # Persist the authenticated user in session state and set the page
            st.session_state["user"] = query_user
            st.session_state["page"] = "Dashboard"
            st.success(f"Welcome back, {query_user.display_name or display_name_input}!")
            return query_user
        else:
            st.error("Invalid username or password.")
    # Provide a cancel/back button to return to the Home page without logging in
    if st.button("Cancel", key="login_cancel_button"):
        st.session_state["page"] = "Home"
    return None


def consumer_dashboard(user: User, session: Session) -> None:
    """Display consumer dashboard with nearest outlets and recommendations.

    The dashboard shows the user's profile picture (if available), display name,
    and favourite drink.  It finds the nearest outlet based on the user's
    stored coordinates.  If the nearest outlet is within 2 km, it displays
    details and predictions using the stock condition and brand presence
    models.  Otherwise it presents an interactive map of all outlets.  Users
    can rate outlets and choose alternative drinks.  The layout uses
    Streamlit columns to organise the profile card and outlet information.
    """
    st.subheader("Consumer Dashboard")
    # Determine what to call the user: prefer display_name, fall back to a generic label
    display_name = getattr(user, "display_name", None) or "User"
    # Provide a way to edit the profile
    if st.button("Edit Profile", key="consumer_edit_profile"):
        st.session_state["page"] = "Edit Profile"
        return
    # Provide a logout button
    if st.button("Logout", key="consumer_logout"):
        st.session_state["user"] = None
        st.session_state["page"] = "Home"
        return
    # Layout: profile card on the left, content on the right
    col1, col2 = st.columns([1, 3])
    with col1:
        # Show profile picture if available
        if hasattr(user, "display_pic") and getattr(user, "display_pic"):
            try:
                st.image(user.display_pic, width=120, caption="", use_container_width=False)
            except Exception:
                pass
        else:
            st.image(Image.new("RGB", (120, 120), color=(200, 200, 200)), width=120)
    with col2:
        st.markdown(f"### {display_name}")
        st.markdown(f"**Favourite Drink:** {user.favourite_drink}")
        # Additional user info (email) if you wish to display
        if getattr(user, "email", None):
            st.markdown(f"{user.email}")
    # Retrieve outlets dataframe and find nearest ones
    outlets_df = get_outlet_dataframe(session)
    if outlets_df.empty:
        st.info("No outlets available yet. Please check back later.")
        return
    nearest = nearest_outlets(
        user.home_latitude, user.home_longitude, outlets_df, k=5
    )
    # If the user is far from any outlet (distance > 2 km), show map of all outlets
    if not nearest.empty and nearest.iloc[0]["distance_km"] > 2.0:
        st.info(
            "You are more than 2 km from the nearest outlet. "
            "Here is a map of all outlets so you can explore."
        )
        fig = px.scatter_map(
            outlets_df,
            lat="latitude",
            lon="longitude",
            color="type",
            hover_name="name",
            hover_data={"brands": True, "distance_km": False},
        )
        st.plotly_chart(fig, width=True)
        # Placeholder for future ordering and messaging features
        st.markdown(
            "*Ordering and messaging features are coming soon! You'll be able to order drinks and chat with outlet owners directly from this map.*"
        )
    else:
        # Show the nearest outlet with details
        top = nearest.iloc[0]
        st.markdown(
            f"The nearest outlet to you is **{top['name']}** ({top['type']}) at "
            f"{top['distance_km']:.2f} km away."
        )
        st.markdown(
            f"**Available brands:** {top['brands']}"
        )
        # Provide a button to patronize this outlet.  Clicking this logs an interaction
        if st.button("Patronize this outlet", key=f"patronize_{top['outlet_id']}"):
            # Record the patronization in the UserInteraction table (simple stub)
            session.add(
                UserInteraction(
                    user_id=user.user_id,
                    outlet_id=int(top["outlet_id"]),
                    brand_id=None,
                    action="patronize",
                )
            )
            session.commit()
            st.success("Thank you for patronizing this outlet! Your visit has been recorded.")
        # Provide a call icon (placeholder)
        st.markdown("☎️ Call outlet owner (coming soon)")
        # Show contact info if the outlet has a display_pic and/or contact field
        # Note: contact_info is not persisted in this schema.  This is a placeholder for future work.
        # st.markdown("☎️ [Call outlet owner](#)")
        # Use the stock condition model (if available) to predict an estimate for this outlet.
        stock_model = load_stock_condition_model()
        if stock_model is not None:
            num_brands = len([b for b in top["brands"].split(",") if b.strip()])
            # Normalise the outlet type for prediction
            normal_type = normalise_outlet_type(top["type"])
            predicted = predict_stock_condition(stock_model, normal_type, num_brands)
            if predicted:
                st.markdown(f"**Predicted stock condition:** {predicted}")
        # Estimate brand availability using trained presence models if available.
        presence_models = load_brand_presence_models()
        fav_drink = user.favourite_drink
        # Compute feature row for the model: we only know the outlet type and number of brands.
        feature_df = pd.DataFrame(
            [
                {
                    "Type_Of_Outlet": normalise_outlet_type(top["type"]),
                    "num_brands_present": num_brands,
                    "num_package_types": 0,
                    "num_display_methods": 0,
                }
            ]
        )
        if fav_drink in presence_models:
            try:
                proba = presence_models[fav_drink].predict_proba(feature_df)[0][1]
                st.markdown(
                    f"Model‑estimated probability that **{fav_drink}** is available here: {proba:.1%}"
                )
            except Exception:
                pass
        st.markdown(
            "*Ordering and messaging features are coming soon! You'll be able to place orders and chat directly with this outlet.*"
        )
        # Recommend alternatives if favourite not available
        brands_in_outlet = [b.strip() for b in top["brands"].split(",") if b.strip()]
        # Count how many distinct brands this outlet stocks.  This will feed into
        # the stock condition and brand presence models.
        num_brands = len(brands_in_outlet)
        if user.favourite_drink not in brands_in_outlet:
            st.markdown(
                f"Sorry, **{user.favourite_drink}** is not stocked here. "
                "Here are some alternatives from similar categories:"
            )
            # Simple category‑based recommendation: look up category of favourite and suggest others
            fav_brand = session.query(Brand).filter_by(name=user.favourite_drink).first()
            if fav_brand:
                # Collect brands in the same category available at this outlet
                alt_brands = []
                for b_name in brands_in_outlet:
                    b_obj = session.query(Brand).filter_by(name=b_name).first()
                    if b_obj and b_obj.category == fav_brand.category:
                        alt_brands.append(b_obj.name)
                if not alt_brands:
                    # If no same-category drinks, show all available as fallbacks
                    alt_brands = brands_in_outlet
                choice = st.radio(
                    "Choose an alternative beverage:", alt_brands, index=0
                )
                if st.button("Select Alternative"):
                    # Log the interaction
                    session.add(
                        UserInteraction(
                            user_id=user.user_id,
                            outlet_id=int(top["outlet_id"]),
                            brand_id=session.query(Brand)
                            .filter_by(name=choice)
                            .first()
                            .brand_id,
                            action="accepted_alternative",
                        )
                    )
                    session.commit()
                    st.success(f"Great! You've selected **{choice}**.")
        else:
            st.markdown(
                f"Good news! Your favourite drink **{user.favourite_drink}** is available here."
            )

        # Allow rating the outlet
        st.markdown("#### Rate this outlet")
        rating = st.slider("Rating (1 = poor, 5 = excellent)", 1, 5, 3)
        comment = st.text_area("Leave a comment (optional)")
        if st.button("Submit Rating"):
            session.add(
                OutletRating(
                    outlet_id=int(top["outlet_id"]),
                    user_id=user.user_id,
                    rating=rating,
                    comment=comment,
                )
            )
            session.commit()
            st.success("Thank you for your feedback!")


def owner_dashboard(user: User, session: Session) -> None:
    """Display a basic dashboard for outlet owners.

    The owner dashboard lists the owner's outlets with their logos (if provided),
    outlet names, types, and stocked brands.  It serves as a starting
    point for future analytics and competitor insights.  Planned features
    such as stock condition forecasts, competitor analysis, ordering and
    messaging are noted but not yet implemented.
    """
    st.subheader("Outlet Owner Dashboard")
    st.markdown(
        "Here you will see analytics about your outlets, competitor outlets nearby, "
        "and stock condition forecasts. These features are under development."
    )
    # Allow owners to edit their profile from the dashboard
    if st.button("Edit Profile", key="owner_edit_profile"):
        st.session_state["page"] = "Edit Profile"
        return
    # Allow owners to add another outlet.  This navigates to the owner sign‑up page.
    if st.button("Add New Outlet", key="owner_add_outlet"):
        st.info("Feature is coming soon !")
        #st.session_state["page"] = "Sign Up"
        # Use a flag in session to preselect the owner role; the sign‑up page will pick up the selection.
        #st.session_state["signup_preselect_owner"] = True
        return
    # Provide a logout button
    if st.button("Logout", key="owner_logout"):
        st.session_state["user"] = None
        st.session_state["page"] = "Home"
        return
    outlets = session.query(Outlet).filter_by(owner_id=user.user_id).all()
    if outlets:
        for o in outlets:
            # Use columns to align logo and text
            logo_col, info_col = st.columns([1, 3])
            with logo_col:
                # Display the logo if present
                if hasattr(o, "display_pic") and getattr(o, "display_pic"):
                    try:
                        st.image(o.display_pic, width=80)
                    except Exception:
                        pass
                else:
                    st.image(Image.new("RGB", (80, 80), color=(220, 220, 220)), width=80)
            with info_col:
                st.markdown(f"### {o.name} ({o.outlet_type})")
                brands = [ob.brand.name for ob in o.brands]
                num_brands = len(brands)
                st.markdown(f"**Brands stocked:** {', '.join(brands) if brands else 'None'}")
                # Placeholder for analytics and competitor analysis
                st.markdown("**Analytics:** Coming soon")
                st.markdown("**Sales & inventory:** Coming soon")
                st.markdown("**Debtor management:** Coming soon")
    else:
        st.info("You have not registered any outlets yet. Use the Sign Up page to add one.")


def edit_profile(user: User, session: Session) -> None:
    """Allow users to edit their profile information.

    Consumers can update their display name, email, favourite drink, password and profile picture.
    Owners can update their display name, email, password, and the name or logo of their first outlet.
    """
    st.subheader("Edit Profile")
    # Provide navigation back to the dashboard
    if st.button("Back to Dashboard", key="edit_back_to_dashboard"):
        st.session_state["page"] = "Dashboard"
        # Immediately display the dashboard and exit this function
        if st.session_state.get("user"):
            # Fetch the latest user record
            refreshed = session.get(User, st.session_state["user"].user_id)
            if refreshed.role == "consumer":
                consumer_dashboard(refreshed, session)
            else:
                owner_dashboard(refreshed, session)
        return
    if user.role == "consumer":
        # Editable fields for consumers
        new_display_name = st.text_input("Display Name", value=user.display_name or "", key="edit_consumer_display_name")
        new_email = st.text_input("Email", value=user.email or "", key="edit_consumer_email")
        # Favourite drink selector
        brands = load_brand_list(session)
        idx = 0
        if user.favourite_drink in brands:
            idx = brands.index(user.favourite_drink)
        new_favourite = st.selectbox("Favourite Drink", brands, index=idx, key="edit_consumer_fav")
        # Password change
        new_password = st.text_input("New Password (optional)", type="password", key="edit_consumer_password")
        confirm_new_password = st.text_input("Confirm New Password", type="password", key="edit_consumer_confirm_password")
        # Profile picture change
        new_file = st.file_uploader("Change profile picture (optional)", type=["jpg", "jpeg", "png"], key="edit_consumer_profile_pic")
        new_pic: Optional[bytes] = None
        if new_file:
            if new_file.size > 1_000_000:
                st.warning("The uploaded file is larger than 1 MB. Please choose a smaller image.")
            else:
                try:
                    new_file.seek(0)
                    data = new_file.getvalue()
                    img = Image.open(io.BytesIO(data))
                    img.thumbnail((300, 300))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    new_pic = buf.getvalue()
                except Exception:
                    st.warning("There was an error processing your image. It will be ignored.")
                    new_pic = None
        if st.button("Save Changes", key="consumer_save_profile"):
            # Validate new passwords
            if new_password and new_password != confirm_new_password:
                st.warning("New passwords do not match.")
            else:
                # Check for duplicate display names (exclude current user)
                existing = session.query(User).filter(User.display_name == new_display_name, User.user_id != user.user_id).first()
                if existing:
                    st.warning("Display name already exists. Please choose another.")
                else:
                    user.display_name = new_display_name
                    user.email = new_email
                    user.favourite_drink = new_favourite
                    if new_password:
                        user.password_hash = hash_password(new_password)
                    if new_pic is not None and hasattr(user, "display_pic"):
                        user.display_pic = new_pic
                    session.commit()
                    st.success("Profile updated.")
    else:
        # Editable fields for owners
        new_display_name = st.text_input("Display Name", value=user.display_name or "", key="edit_owner_display_name")
        new_email = st.text_input("Email", value=user.email or "", key="edit_owner_email")
        new_password = st.text_input("New Password (optional)", type="password", key="edit_owner_password")
        confirm_owner_password = st.text_input("Confirm New Password", type="password", key="edit_owner_confirm_password")
        # Retrieve the first outlet for editing
        outlet = session.query(Outlet).filter_by(owner_id=user.user_id).first()
        new_outlet_name = None
        new_logo: Optional[bytes] = None
        new_selected_brands: Optional[List[str]] = None
        if outlet:
            # Allow editing the business name
            new_outlet_name = st.text_input("Business Name", value=outlet.name or "", key="edit_owner_business_name")
            # Allow changing the logo
            logo_file = st.file_uploader("Change business logo (optional)", type=["jpg", "jpeg", "png"], key="edit_owner_logo")
            if logo_file:
                if logo_file.size > 1_000_000:
                    st.warning("The uploaded logo is larger than 1 MB. Please choose a smaller file.")
                else:
                    try:
                        logo_file.seek(0)
                        data = logo_file.getvalue()
                        img = Image.open(io.BytesIO(data))
                        img.thumbnail((300, 300))
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG")
                        new_logo = buf.getvalue()
                    except Exception:
                        st.warning("There was an error processing your logo. It will be ignored.")
                        new_logo = None
            # Allow updating the brands stocked at this outlet
            brand_options = load_brand_list(session)
            current_brands = [ob.brand.name for ob in outlet.brands]
            new_selected_brands = st.multiselect(
                "Select brands you stock", brand_options, default=current_brands, key="edit_owner_brands"
            )
        if st.button("Save Changes", key="owner_save_profile"):
            if new_password and new_password != confirm_owner_password:
                st.warning("New passwords do not match.")
            else:
                # Check for duplicate display names (exclude current user)
                existing = session.query(User).filter(User.display_name == new_display_name, User.user_id != user.user_id).first()
                if existing:
                    st.warning("Display name already exists. Please choose another.")
                else:
                    user.display_name = new_display_name
                    user.email = new_email
                    if new_password:
                        user.password_hash = hash_password(new_password)
                    # Update outlet details if present
                    if outlet:
                        if new_outlet_name is not None:
                            outlet.name = new_outlet_name
                        if new_logo is not None and hasattr(outlet, "display_pic"):
                            outlet.display_pic = new_logo
                        # Update brands if a selection was provided
                        if new_selected_brands is not None:
                            # Remove existing brand associations for this outlet
                            session.query(OutletBrand).filter_by(outlet_id=outlet.outlet_id).delete()
                            # Add new associations
                            for b_name in new_selected_brands:
                                brand = session.query(Brand).filter_by(name=b_name).first()
                                if brand:
                                    session.add(
                                        OutletBrand(
                                            outlet_id=outlet.outlet_id,
                                            brand_id=brand.brand_id,
                                            stock_status="Well stocked",
                                        )
                                    )
                    session.commit()
                    st.success("Profile updated.")


def explore_page(user: Optional[User], session: Session) -> None:
    """Provide exploration and discovery features.

    This page shows trending brands across all outlets and lets users search for outlets
    stocking a specific brand.  When a user is logged in, the distances shown in the
    search results are computed from the user's registered home coordinates; otherwise
    distances are omitted.
    """
    st.subheader("Explore Drinks & Outlets")
    # Trending brands: count the number of outlets stocking each brand
    # We use an aggregation query on the OutletBrand table joined with Brand
    brand_counts = (
        session.query(Brand.name, func.count(OutletBrand.outlet_id))
        .join(OutletBrand, Brand.brand_id == OutletBrand.brand_id)
        .group_by(Brand.name)
        .order_by(desc(func.count(OutletBrand.outlet_id)))
        .all()
    )
    if brand_counts:
        top5 = brand_counts[:5]
        names = [b[0] for b in top5]
        counts = [b[1] for b in top5]
        st.markdown("#### Top 5 Trending Brands")
        # Use a bar chart for better phone compatibility
        trending_df = pd.DataFrame({"Brand": names, "Outlets": counts})
        st.bar_chart(trending_df.set_index("Brand"))
    else:
        st.info("No outlet data available to compute trending brands.")
    # Brand search: allow users to select a brand and see nearby outlets
    st.markdown("---")
    st.markdown("#### Find Outlets by Brand")
    brand_options = load_brand_list(session)
    if not brand_options:
        st.info("No brands available in the database.")
        return
    selected_brand = st.selectbox("Select a brand to search for outlets", brand_options)
    # Retrieve outlets stocking the selected brand
    # We join OutletBrand and Outlet to filter outlets by brand
    outlet_rows = (
        session.query(Outlet)
        .join(OutletBrand, Outlet.outlet_id == OutletBrand.outlet_id)
        .join(Brand, Brand.brand_id == OutletBrand.brand_id)
        .filter(Brand.name == selected_brand)
        .all()
    )
    if outlet_rows:
        # Build a DataFrame with name, type, brands and, if the user is logged in, distance
        records = []
        for o in outlet_rows:
            entry = {
                "Outlet Name": o.name,
                "Type": o.outlet_type,
            }
            # Compute distance only if user provided home coordinates
            if user is not None and getattr(user, "home_latitude", None) is not None:
                dist = haversine(user.home_latitude, user.home_longitude, o.latitude, o.longitude)
                entry["Distance (km)"] = round(dist, 2)
            records.append(entry)
        results_df = pd.DataFrame(records)
        st.write(results_df)
    else:
        st.info(f"No outlets currently stock {selected_brand}.")


def settings_page() -> None:
    """Display application settings.

    Currently this page provides a placeholder for selecting a light or dark theme.
    Due to Streamlit limitations in this environment, the selection is saved in
    ``st.session_state['theme']`` but does not automatically update the app
    appearance.  You can extend this function to apply CSS or use Streamlit's
    built‑in theme configuration when deploying.
    """
    st.subheader("Settings")
    # Retrieve stored theme or default to Light
    current = st.session_state.get("theme", "Light")
    theme_choice = st.selectbox("Select theme", ["Light", "Dark"], index=0 if current == "Light" else 1)
    # Save the selection in session state
    st.session_state["theme"] = theme_choice
    st.info("Theme preference saved. Theme changes will take effect after refresh or in a supported environment.")


def main() -> None:
    st.set_page_config(page_title="The Soda Spot App", layout="wide")
    st.title("Welcome to the Soda Spot Experience")
    st.markdown(
        "Use this application to find your favourite drinks at nearby outlets and "
        "get alternative recommendations. Outlet owners can also manage their stock and get real time feedback from customers as well as competitor analysis and market insights."
    )

    session = SessionLocal()
    try:
        # Initialise session state
        if "page" not in st.session_state:
            st.session_state["page"] = "Home"
        if "user" not in st.session_state:
            st.session_state["user"] = None

        # Dynamically populate sidebar options based on authentication state
        # When no user is logged in, offer Home, Sign Up and Login pages.
        # When a user is logged in, hide Sign Up/Login and instead show the Dashboard and Edit Profile pages.
        if st.session_state.get("user") is None:
            pages = ["Home", "Sign Up", "Login"]
        else:
            # Logged‑in users can explore, edit their profile and see settings
            pages = ["Dashboard", "Explore", "Edit Profile", "Settings"]
        # Show the sidebar with the current page preselected
        sidebar_selection = st.sidebar.selectbox(
            "Select a page",
            pages,
            index=pages.index(st.session_state["page"]) if st.session_state["page"] in pages else 0,
        )
        # If a different page is selected from the sidebar, update session state
        if sidebar_selection != st.session_state["page"]:
            st.session_state["page"] = sidebar_selection
        current_page = st.session_state["page"]

        # If a user is logged in, provide a logout button in the sidebar
        if st.session_state.get("user") is not None:
            if st.sidebar.button("Logout", key="sidebar_logout"):
                st.session_state["user"] = None
                st.session_state["page"] = "Home"
                # After logging out, the next interaction will show the Home page

        # Render Home page
        if current_page == "Home":
            st.markdown(
                "Welcome! Please use the sidebar or the buttons below to sign up or log in. "
                "If you're a new user, choose Sign Up to create an account. "
                "Returning users can go straight to Login and then the Dashboard page."
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create account"):
                    st.session_state["page"] = "Sign Up"
            with col2:
                if st.button("Login"):
                    st.session_state["page"] = "Login"

        elif current_page == "Sign Up":
            # Determine if we should preselect the owner role (e.g. when adding an outlet from the owner dashboard)
            default_role = "consumer"
            if st.session_state.get("signup_preselect_owner"):
                default_role = "owner"
                # Reset the flag so subsequent sign‑ups start at the default
                st.session_state["signup_preselect_owner"] = False
            role = st.radio("I am a...", ["consumer", "owner"], horizontal=True, index=1 if default_role == "owner" else 0)
            if role == "consumer":
                register_consumer(session)
            else:
                register_owner(session)

        elif current_page == "Login":
            # Render the login form and handle authentication
            user = login_user(session)
            # If authentication was successful, user and page will be set in session state
            if st.session_state.get("user") and st.session_state.get("page") == "Dashboard":
                # Immediately display the appropriate dashboard without waiting for a rerun
                refreshed_user = session.get(User, st.session_state["user"].user_id)
                if refreshed_user.role == "consumer":
                    consumer_dashboard(refreshed_user, session)
                else:
                    owner_dashboard(refreshed_user, session)

        elif current_page == "Dashboard":
            if not st.session_state["user"]:
                st.info("Please log in to access your dashboard.")
            else:
                # Refresh user record from the database
                user = session.get(User, st.session_state["user"].user_id)
                if user.role == "consumer":
                    consumer_dashboard(user, session)
                else:
                    owner_dashboard(user, session)

        elif current_page == "Explore":
            # Only allow explore if logged in
            if not st.session_state["user"]:
                st.info("Please log in to explore outlets and drinks.")
            else:
                user = session.get(User, st.session_state["user"].user_id)
                explore_page(user, session)

        elif current_page == "Settings":
            # Settings page: no database needed, but only accessible when logged in
            if not st.session_state["user"]:
                st.info("Please log in to access settings.")
            else:
                settings_page()

        elif current_page == "Edit Profile":
            # Only allow editing if the user is logged in
            if not st.session_state["user"]:
                st.info("Please log in to edit your profile.")
            else:
                user = session.get(User, st.session_state["user"].user_id)
                edit_profile(user, session)

    finally:
        session.close()


if __name__ == "__main__":
    main()
