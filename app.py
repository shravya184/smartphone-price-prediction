import streamlit as st
import pandas as pd
import numpy as np
import pickle


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="A Supervised Machine Learning Approach for Smartphone Price Prediction Using Regression Models",
    page_icon="📱",
    layout="wide"
)


# ============================================================
# LOAD MODEL BUNDLE
# ============================================================

@st.cache_resource
def load_model():

    with open("smartphone_price_model.pkl", "rb") as f:
        bundle = pickle.load(f)

    return bundle


try:
    bundle = load_model()

    model = bundle["model"]
    model_name = bundle["model_name"]

    numeric_imputer = bundle["numeric_imputer"]
    categorical_imputer = bundle["categorical_imputer"]
    encoder = bundle["encoder"]
    scaler = bundle["scaler"]

    numeric_columns = bundle["numeric_columns"]
    categorical_columns = bundle["categorical_columns"]

    use_log_transform = bundle["use_log_transform"]

except Exception as e:

    st.error("❌ Failed to load the trained model.")
    st.exception(e)
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📱 Smartphone Price Prediction")

st.write(
    "Enter the smartphone specifications below to estimate its price."
)

st.info(
    f"🤖 Loaded Model: **{model_name}**"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📱 Smartphone Price Predictor")

st.sidebar.write(
    "Enter technical specifications and click **Predict Price**."
)

st.sidebar.markdown("---")

st.sidebar.write(
    "The prediction uses the same preprocessing and feature "
    "engineering used during model training."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📋 Smartphone Specifications")


# ------------------------------------------------------------
# BASIC INFORMATION
# ------------------------------------------------------------

st.subheader("1️⃣ Basic Information")

col1, col2, col3 = st.columns(3)


with col1:

    brand_options = list(encoder.categories_[categorical_columns.index("brand_name")])

    brand_name = st.selectbox(
        "Brand",
        options=brand_options
    )


with col2:

    avg_rating = st.number_input(
        "Average Rating",
        min_value=0.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )


with col3:

    os_options = list(
        encoder.categories_[categorical_columns.index("os")]
    )

    os = st.selectbox(
        "Operating System",
        options=os_options
    )


# ============================================================
# CONNECTIVITY
# ============================================================

st.subheader("2️⃣ Connectivity & Charging")

col1, col2, col3, col4 = st.columns(4)


with col1:

    five_g = st.selectbox(
        "5G Support",
        options=["Yes", "No"]
    )

    five_g_value = 1 if five_g == "Yes" else 0


with col2:

    fast_charging_available = st.selectbox(
        "Fast Charging Available",
        options=["Yes", "No"]
    )

    fast_charging_available_value = (
        1 if fast_charging_available == "Yes" else 0
    )


with col3:

    fast_charging = st.number_input(
        "Fast Charging (W)",
        min_value=0.0,
        max_value=300.0,
        value=33.0,
        step=1.0
    )


with col4:

    extended_memory_available = st.selectbox(
        "Extended Memory Available",
        options=["Yes", "No"]
    )

    extended_memory_available_value = (
        1 if extended_memory_available == "Yes" else 0
    )


# ============================================================
# PROCESSOR
# ============================================================

st.subheader("3️⃣ Processor")

col1, col2, col3 = st.columns(3)


with col1:

    processor_options = list(
        encoder.categories_[
            categorical_columns.index("processor_brand")
        ]
    )

    processor_brand = st.selectbox(
        "Processor Brand",
        options=processor_options
    )


with col2:

    num_cores = st.number_input(
        "Number of Cores",
        min_value=1,
        max_value=24,
        value=8,
        step=1
    )


with col3:

    processor_speed = st.number_input(
        "Processor Speed (GHz)",
        min_value=0.1,
        max_value=10.0,
        value=2.5,
        step=0.1
    )


# ============================================================
# MEMORY
# ============================================================

st.subheader("4️⃣ Memory & Storage")

col1, col2, col3 = st.columns(3)


with col1:

    ram_capacity = st.number_input(
        "RAM (GB)",
        min_value=1,
        max_value=64,
        value=8,
        step=1
    )


with col2:

    internal_memory = st.number_input(
        "Internal Storage (GB)",
        min_value=1,
        max_value=2048,
        value=128,
        step=1
    )


with col3:

    battery_capacity = st.number_input(
        "Battery Capacity (mAh)",
        min_value=500.0,
        max_value=20000.0,
        value=5000.0,
        step=100.0
    )


# ============================================================
# DISPLAY
# ============================================================

st.subheader("5️⃣ Display")

col1, col2, col3 = st.columns(3)


with col1:

    screen_size = st.number_input(
        "Screen Size (inches)",
        min_value=2.0,
        max_value=15.0,
        value=6.5,
        step=0.1
    )


with col2:

    refresh_rate = st.number_input(
        "Refresh Rate (Hz)",
        min_value=30,
        max_value=240,
        value=120,
        step=1
    )


with col3:

    resolution_height = st.number_input(
        "Resolution Height",
        min_value=100,
        max_value=5000,
        value=2400,
        step=1
    )


# ============================================================
# CAMERA
# ============================================================

st.subheader("6️⃣ Camera")

col1, col2, col3 = st.columns(3)


with col1:

    primary_camera_rear = st.number_input(
        "Primary Rear Camera (MP)",
        min_value=0.0,
        max_value=300.0,
        value=50.0,
        step=1.0
    )


with col2:

    primary_camera_front = st.number_input(
        "Primary Front Camera (MP)",
        min_value=0.0,
        max_value=100.0,
        value=16.0,
        step=1.0
    )


with col3:

    num_rear_cameras = st.number_input(
        "Number of Rear Cameras",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )


# ============================================================
# RESOLUTION
# ============================================================

st.subheader("7️⃣ Display Resolution")

resolution_width = st.number_input(
    "Resolution Width",
    min_value=100,
    max_value=5000,
    value=1080,
    step=1
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Smartphone Price",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # 1. CREATE RAW INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "brand_name": brand_name,
            "avg_rating": avg_rating,
            "5G_or_not": five_g_value,
            "processor_brand": processor_brand,
            "num_cores": num_cores,
            "processor_speed": processor_speed,
            "battery_capacity": battery_capacity,
            "fast_charging_available":
                fast_charging_available_value,
            "fast_charging": fast_charging,
            "ram_capacity": ram_capacity,
            "internal_memory": internal_memory,
            "screen_size": screen_size,
            "refresh_rate": refresh_rate,
            "num_rear_cameras": num_rear_cameras,
            "os": os,
            "primary_camera_rear": primary_camera_rear,
            "primary_camera_front": primary_camera_front,
            "extended_memory_available":
                extended_memory_available_value,
            "resolution_height": resolution_height,
            "resolution_width": resolution_width

        }])


        # ----------------------------------------------------
        # 2. FEATURE ENGINEERING
        # ----------------------------------------------------

        input_data["performance_score"] = (
            input_data["ram_capacity"]
            *
            input_data["processor_speed"]
        )


        input_data["camera_score"] = (
            input_data["primary_camera_rear"]
            +
            input_data["primary_camera_front"]
        )


        input_data["display_score"] = (
            input_data["resolution_height"]
            *
            input_data["resolution_width"]
            *
            input_data["refresh_rate"]
        )


        input_data["storage_to_ram"] = (
            input_data["internal_memory"]
            /
            input_data["ram_capacity"]
        )


        # ----------------------------------------------------
        # 3. ENSURE COLUMN ORDER
        # ----------------------------------------------------

        input_numeric = input_data[numeric_columns].copy()

        input_categorical = input_data[categorical_columns].copy()


        # ----------------------------------------------------
        # 4. NUMERICAL IMPUTATION
        # ----------------------------------------------------

        input_numeric_imputed = (
            numeric_imputer.transform(input_numeric)
        )


        # ----------------------------------------------------
        # 5. CATEGORICAL IMPUTATION
        # ----------------------------------------------------

        input_categorical_imputed = (
            categorical_imputer.transform(input_categorical)
        )


        # ----------------------------------------------------
        # 6. ONE-HOT ENCODING
        # ----------------------------------------------------

        input_categorical_encoded = (
            encoder.transform(input_categorical_imputed)
        )


        # ----------------------------------------------------
        # 7. STANDARD SCALING
        # ----------------------------------------------------

        input_numeric_scaled = (
            scaler.transform(input_numeric_imputed)
        )


        # ----------------------------------------------------
        # 8. COMBINE FEATURES
        # ----------------------------------------------------

        input_final = np.hstack([
            input_numeric_scaled,
            input_categorical_encoded
        ])


        # ----------------------------------------------------
        # 9. MODEL PREDICTION
        # ----------------------------------------------------

        prediction_log = model.predict(input_final)


        # ----------------------------------------------------
        # 10. CONVERT LOG PRICE BACK TO ORIGINAL PRICE
        # ----------------------------------------------------

        if use_log_transform:

            predicted_price = np.expm1(
                prediction_log[0]
            )

        else:

            predicted_price = prediction_log[0]


        predicted_price = max(0, predicted_price)


        # ----------------------------------------------------
        # 11. DISPLAY RESULT
        # ----------------------------------------------------

        st.success("✅ Prediction completed successfully!")

        st.markdown(
            f"""
            ## 💰 Estimated Smartphone Price

            # ₹{predicted_price:,.0f}
            """
        )


        # ----------------------------------------------------
        # 12. INPUT SUMMARY
        # ----------------------------------------------------

        with st.expander("🔍 View Input Details"):

            st.dataframe(
                input_data.T.rename(
                    columns={0: "Value"}
                ),
                use_container_width=True
            )


        # ----------------------------------------------------
        # 13. MODEL INFORMATION
        # ----------------------------------------------------

        st.info(
            f"""
            **Model Used:** {model_name}

            **Prediction Method:** Log-transformed regression

            **Feature Engineering:** 4 engineered features

            **Preprocessing:** Imputation + One-Hot Encoding + StandardScaler
            """
        )


    except Exception as e:

        st.error("❌ Prediction failed.")

        st.exception(e)