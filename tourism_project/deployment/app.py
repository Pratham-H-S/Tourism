import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import pickle

st.set_page_config(page_title="Tourism Package Predictor", page_icon="🏝️", layout="wide")

# ============================================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================================

@st.cache_resource
def load_model_and_preprocessors():
    """Load model, scaler, and encoders from HuggingFace"""
    try:
        # Download model
        model_path = hf_hub_download(
            repo_id="DD009/tourism-package-model",
            filename="best_tourism_model.joblib"
        )
        model = joblib.load(model_path)
        
        # Download scaler
        scaler_path = hf_hub_download(
            repo_id="DD009/tourism-package-model",
            filename="scaler.pkl"
        )
        scaler = joblib.load(scaler_path)
        
        # Download label encoders
        encoders_path = hf_hub_download(
            repo_id="DD009/tourism-package-model",
            filename="label_encoders.pkl"
        )
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        
        return model, scaler, encoders
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the model is uploaded to HuggingFace")
        return None, None, None

model, scaler, encoders = load_model_and_preprocessors()

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🏝️ Wellness Tourism Package Purchase Predictor")
st.markdown("""
### Predict Customer Purchase Likelihood
This application predicts whether a customer will purchase the **Wellness Tourism Package** 
based on their demographics, travel preferences, and sales interaction data.
""")

st.markdown("---")

if model is None:
    st.error("❌ Model not loaded. Please check HuggingFace repository.")
    st.stop()

# ============================================================================
# INPUT FORM
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Demographics")
    
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    marital_status = st.selectbox(
        "Marital Status", 
        ["Single", "Married", "Divorced", "Unmarried"]
    )
    
    occupation = st.selectbox(
        "Occupation", 
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    
    designation = st.selectbox(
        "Designation", 
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    
    monthly_income = st.number_input(
        "Monthly Income (₹)", 
        min_value=0, 
        max_value=200000, 
        value=50000, 
        step=1000
    )

with col2:
    st.subheader("✈️ Travel Preferences")
    
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    
    type_of_contact = st.selectbox(
        "Type of Contact", 
        ["Self Enquiry", "Company Invited"]
    )
    
    num_persons = st.number_input(
        "Number of Persons Visiting", 
        min_value=1, 
        max_value=10, 
        value=2, 
        step=1
    )
    
    num_children = st.number_input(
        "Number of Children (below 5)", 
        min_value=0, 
        max_value=5, 
        value=0, 
        step=1
    )
    
    property_star = st.selectbox(
        "Preferred Property Star Rating", 
        [3.0, 4.0, 5.0]
    )
    
    num_trips = st.number_input(
        "Number of Trips Per Year", 
        min_value=0, 
        max_value=20, 
        value=2, 
        step=1
    )
    
    passport = st.selectbox("Has Valid Passport", ["Yes", "No"])
    passport_val = 1 if passport == "Yes" else 0
    
    own_car = st.selectbox("Owns Car", ["Yes", "No"])
    own_car_val = 1 if own_car == "Yes" else 0

st.markdown("---")

st.subheader("💼 Sales Interaction")

col3, col4 = st.columns(2)

with col3:
    product_pitched = st.selectbox(
        "Product Pitched", 
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    
    pitch_satisfaction = st.slider(
        "Pitch Satisfaction Score", 
        min_value=1, 
        max_value=5, 
        value=3
    )

with col4:
    num_followups = st.number_input(
        "Number of Follow-ups", 
        min_value=0, 
        max_value=10, 
        value=3, 
        step=1
    )
    
    duration_pitch = st.number_input(
        "Duration of Pitch (minutes)", 
        min_value=0, 
        max_value=60, 
        value=15, 
        step=1
    )

st.markdown("---")

# ============================================================================
# PREDICTION
# ============================================================================

if st.button("🔮 Predict Purchase Probability", type="primary", use_container_width=True):
    
    # Create input dataframe with exact column names from training
    input_data = pd.DataFrame([{
        'Age': age,
        'TypeofContact': type_of_contact,
        'CityTier': city_tier,
        'DurationOfPitch': duration_pitch,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': num_persons,
        'NumberOfFollowups': num_followups,
        'ProductPitched': product_pitched,
        'PreferredPropertyStar': property_star,
        'MaritalStatus': marital_status,
        'NumberOfTrips': num_trips,
        'Passport': passport_val,
        'PitchSatisfactionScore': pitch_satisfaction,
        'OwnCar': own_car_val,
        'NumberOfChildrenVisiting': num_children,
        'Designation': designation,
        'MonthlyIncome': monthly_income
    }])
    
    try:
        # Encode categorical variables
        for col in input_data.select_dtypes(include=['object']).columns:
            if col in encoders:
                try:
                    input_data[col] = encoders[col].transform(input_data[col].astype(str))
                except:
                    # Handle unseen categories
                    input_data[col] = 0
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            if prediction == 1:
                st.success("### ✅ Will Purchase")
                st.markdown("**Prediction:** Customer is likely to buy the package")
            else:
                st.error("### ❌ Will Not Purchase")
                st.markdown("**Prediction:** Customer is unlikely to buy the package")
        
        with col_r2:
            purchase_prob = probability[1] * 100
            st.metric(
                "Purchase Probability", 
                f"{purchase_prob:.1f}%",
                delta=None
            )
        
        with col_r3:
            confidence = max(probability) * 100
            st.metric(
                "Model Confidence", 
                f"{confidence:.1f}%",
                delta=None
            )
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendation")
        
        if probability[1] >= 0.7:
            st.success("""
            **🌟 High Priority Lead**
            - Probability: Very High (>70%)
            - Action: Contact immediately with personalized offer
            - Follow-up: Within 24 hours
            - Offer: Premium package with special discount
            """)
        elif probability[1] >= 0.5:
            st.warning("""
            **⚡ Medium Priority Lead**
            - Probability: Moderate (50-70%)
            - Action: Schedule follow-up call within 2-3 days
            - Follow-up: Regular contact
            - Offer: Standard package with competitive pricing
            """)
        elif probability[1] >= 0.3:
            st.info("""
            **📧 Low Priority Lead**
            - Probability: Low (30-50%)
            - Action: Add to email nurture campaign
            - Follow-up: Monthly newsletters
            - Offer: Budget-friendly options
            """)
        else:
            st.warning("""
            **🔍 Re-evaluate Approach**
            - Probability: Very Low (<30%)
            - Action: May need different package or timing
            - Follow-up: Quarterly check-in
            - Offer: Explore alternative travel options
            """)
        
        # Display probability breakdown
        st.markdown("---")
        st.subheader("📈 Probability Breakdown")
        
        prob_df = pd.DataFrame({
            'Outcome': ['Will Not Purchase', 'Will Purchase'],
            'Probability': [probability[0] * 100, probability[1] * 100]
        })
        
        st.bar_chart(prob_df.set_index('Outcome'))
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.info("Please ensure all preprocessing files are uploaded correctly")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 📝 About
This predictive model uses **XGBoost** trained on historical customer data to predict 
the likelihood of purchasing the Wellness Tourism Package. The model considers:
- Customer demographics (age, income, occupation)
- Travel preferences (city tier, property rating, trips per year)
- Sales interaction data (pitch satisfaction, follow-ups, duration)

**Model Performance:**
- F1-Score: ~85-90%
- ROC-AUC: ~90-95%
- Accuracy: ~85-90%

*Developed as part of MLOps Pipeline Project*
""")
