import pandas as pd
import datetime
import xgboost as xgb
import streamlit as st


def main():
    html_temp = """
    <h1 style='text-align:center;'>Car Price Prediction</h1>
    """

    # Load Model
    model = xgb.XGBRegressor()
    model.load_model("xgb_model.json")

    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("### This app will help you predict your car selling price.")

    # Present Price
    p1 = st.number_input(
        "Please enter ex-showroom price (In Lakhs)",
        min_value=2.5,
        max_value=25.0,
        step=1.0,
    )

    # Kilometers Driven
    p2 = st.number_input(
        "Please enter car driven (In Kilometers)",
        min_value=100,
        max_value=500000,
        step=100,
    )

    # Fuel Type
    s1 = st.selectbox("Select Fuel Type", ("Petrol", "Diesel", "CNG"))

    if s1 == "Petrol":
        p3 = 0
    elif s1 == "Diesel":
        p3 = 1
    else:
        p3 = 2

    # Seller Type
    s2 = st.selectbox("Select Seller Type", ("Dealer", "Individual"))

    if s2 == "Dealer":
        p4 = 0
    else:
        p4 = 1

    # Transmission
    s3 = st.selectbox("Select Transmission", ("Manual", "Automatic"))

    if s3 == "Manual":
        p5 = 0
    else:
        p5 = 1

    # Owner
    p6 = st.slider("How many owners", 0, 3)

    # Car Age
    current_year = datetime.datetime.now().year

    years = st.number_input(
        "Car Purchased Year",
        min_value=1990,
        max_value=current_year,
        step=1,
    )

    p7 = current_year - years

    # Create DataFrame
    data_new = pd.DataFrame({
        "Present_Price": [p1],
        "Kms_Driven": [p2],
        "Fuel_Type": [p3],
        "Seller_Type": [p4],
        "Transmission": [p5],
        "Owner": [p6],
        "Age": [p7],
    })

    # Prediction
    if st.button("Predict"):
        pred = model.predict(data_new)
        st.success(f"You can sell your car at {pred[0]:.2f} Lakhs")


if __name__ == "__main__":
    main()