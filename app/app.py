import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(__file__),"..", "model")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    feature_types = joblib.load(os.path.join(MODEL_DIR, "feature_types.pkl"))
    return model, encoders, feature_columns, feature_types


def build_input_form(feature_columns, feature_types, encoders):
    """Renders one widget per feature column and returns a dict of raw values."""
    categorical_features = set(feature_types["categorical_features"])
    values = {}

    st.subheader("Building details")
    cols = st.columns(2)
    for i, col_name in enumerate(feature_columns):
        target = cols[i % 2]
        with target:
            if col_name in categorical_features:
                options = list(encoders[col_name].classes_)
                values[col_name] = st.selectbox(col_name, options, key=col_name)
            else:
                # Realistic default/range per column (from the training data),
                # instead of a blind 0.0 default. Without this, geo_level_*, age,
                # area_percentage and height_percentage — which together drive
                # ~68% of the model's decision — always got fed as 0, so the
                # prediction barely moved no matter what other fields were changed.
                numeric_defaults = {
                    "geo_level_1_id": (0, 30, 10),
                    "geo_level_2_id": (0, 1427, 500),
                    "geo_level_3_id": (0, 12567, 5000),
                    "count_floors_pre_eq": (1, 9, 2),
                    "age": (0, 995, 20),
                    "area_percentage": (1, 100, 8),
                    "height_percentage": (1, 32, 5),
                    "count_families": (0, 10, 1),
                }
                lo, hi, default = numeric_defaults.get(col_name, (0, 100, 0))
                values[col_name] = st.number_input(
                    col_name, min_value=lo, max_value=hi, value=default,
                    step=1, key=col_name
                )
    return values


def predict(model, encoders, feature_columns, raw_values):
    row = {}
    for col_name in feature_columns:
        val = raw_values[col_name]
        if col_name in encoders:
            le = encoders[col_name]
            val = le.transform([val])[0]
        row[col_name] = val

    X_input = pd.DataFrame([row], columns=feature_columns)
    pred = model.predict(X_input)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]
        proba = dict(zip(model.classes_, proba))
    return pred, proba


def main():
    st.set_page_config(page_title="Earthquake Damage Prediction", page_icon="🏚️")
    st.title("🏚️ Earthquake Damage Grade Prediction")
    st.caption(
        "Predicts the likely damage grade of a building using a Random Forest "
        "model trained on structural + location features."
    )

    try:
        model, encoders, feature_columns, feature_types = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model files not found in ./model/. Run the notebook "
            "(notebook/Earthquake_Damage_Prediction.ipynb) first — its last "
            "cell saves model.pkl, encoders.pkl, feature_columns.pkl and "
            "feature_types.pkl into this model/ folder."
        )
        return

    raw_values = build_input_form(feature_columns, feature_types, encoders)

    if st.button("Predict Damage Grade", type="primary"):
        pred, proba = predict(model, encoders, feature_columns, raw_values)
        st.success(f"Predicted damage grade: **{pred}**")
        if proba:
            st.write("Class probabilities:")
            st.bar_chart(pd.Series(proba))


if __name__ == "__main__":
    main()
