import streamlit as st
import pandas as pd
import pickle


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    with open(
        "customer_churn_LGBMC.pkl",
        "rb"
    ) as file:

        model = pickle.load(file)

    return model


try:

    model = load_model()

except Exception as e:

    st.error("❌ Model could not be loaded")

    st.code(str(e))

    st.stop()


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="title">'
    '📉 Customer Churn Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict whether a customer is likely to churn'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📌 About")

    st.write(
        """
        This application uses a
        Gradient Boosting Classifier
        to predict customer churn.
        """
    )

    st.info(
        "CustomerID is not used for prediction."
    )


# =====================================================
# CUSTOMER FORM
# =====================================================

with st.form("customer_churn_form"):

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)


    # =================================================
    # COLUMN 1
    # =================================================

    with col1:

        age = st.number_input(
            "Age",
            min_value=18.0,
            max_value=100.0,
            value=30.0,
            step=1.0
        )

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )

        tenure = st.number_input(
            "Tenure",
            min_value=0.0,
            value=12.0,
            step=1.0
        )

        usage_frequency = st.number_input(
            "Usage Frequency",
            min_value=0.0,
            value=10.0,
            step=1.0
        )


    # =================================================
    # COLUMN 2
    # =================================================

    with col2:

        support_calls = st.number_input(
            "Support Calls",
            min_value=0.0,
            value=2.0,
            step=1.0
        )

        payment_delay = st.number_input(
            "Payment Delay",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        subscription_type = st.selectbox(
            "Subscription Type",
            [
                "Basic",
                "Premium",
                "Standard"
            ]
        )


    # =================================================
    # COLUMN 3
    # =================================================

    with col3:

        contract_length = st.selectbox(
            "Contract Length",
            [
                "Annual",
                "Monthly",
                "Quarterly"
            ]
        )

        total_spend = st.number_input(
            "Total Spend",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        last_interaction = st.number_input(
            "Last Interaction",
            min_value=0.0,
            value=10.0,
            step=1.0
        )


    st.divider()


    # =================================================
    # BUTTON
    # =================================================

    predict_button = st.form_submit_button(
        "🔍 Predict Customer Churn",
        use_container_width=True
    )


# =====================================================
# PREDICTION
# =====================================================

if predict_button:

    try:

        # =================================================
        # ORIGINAL INPUT
        # =================================================

        input_data = pd.DataFrame([{

            "Age": age,

            "Tenure": tenure,

            "Usage Frequency": usage_frequency,

            "Support Calls": support_calls,

            "Payment Delay": payment_delay,

            "Total Spend": total_spend,

            "Last Interaction": last_interaction,

            "Gender": gender,

            "Subscription Type": subscription_type,

            "Contract Length": contract_length

        }])


        # =================================================
        # CATEGORICAL COLUMNS
        # =================================================

        cat_cols = [
            "Gender",
            "Subscription Type",
            "Contract Length"
        ]


        # =================================================
        # ONE-HOT ENCODING
        # =================================================

        input_encoded = pd.get_dummies(
            input_data,
            columns=cat_cols,
            drop_first=True
        )


        # =================================================
        # EXACT TRAINING COLUMNS
        # =================================================

        model_columns = [
            "Age",
            "Tenure",
            "Usage Frequency",
            "Support Calls",
            "Payment Delay",
            "Total Spend",
            "Last Interaction",
            "Gender_Male",
            "Subscription Type_Premium",
            "Subscription Type_Standard",
            "Contract Length_Monthly",
            "Contract Length_Quarterly"
        ]


        # =================================================
        # MATCH COLUMN ORDER
        # =================================================

        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )


        # =================================================
        # CONVERT TO FLOAT
        # =================================================

        input_encoded = input_encoded.astype(float)


        # =================================================
        # CHECK FEATURES
        # =================================================

        if input_encoded.shape[1] != 12:

            st.error(
                f"❌ Model input has "
                f"{input_encoded.shape[1]} features."
            )

            st.stop()


        # =================================================
        # PREDICTION
        # =================================================

        prediction = model.predict(
            input_encoded
        )[0]


        # =================================================
        # PROBABILITY
        # =================================================

        probability = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probability = model.predict_proba(
                input_encoded
            )[0][1]


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader(
            "📊 Prediction Result"
        )


        if prediction == 1:

            st.error(
                "## ⚠️ Customer is likely to CHURN"
            )

            st.warning(
                "Consider taking customer "
                "retention actions."
            )

        else:

            st.success(
                "## ✅ Customer is likely to STAY"
            )

            st.info(
                "This customer currently appears "
                "less likely to churn."
            )


        # =================================================
        # PROBABILITY
        # =================================================

        if probability is not None:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )


        # =================================================
        # SHOW CUSTOMER DATA
        # =================================================

        with st.expander(
            "📋 View Customer Information"
        ):

            st.dataframe(
                input_data,
                use_container_width=True
            )


        # =================================================
        # SHOW ENCODED DATA
        # =================================================

        with st.expander(
            "🔢 View Model Input"
        ):

            st.write(
                f"Model features: "
                f"{input_encoded.shape[1]}"
            )

            st.dataframe(
                input_encoded,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "❌ Prediction Error"
        )

        st.code(
            str(e)
        )