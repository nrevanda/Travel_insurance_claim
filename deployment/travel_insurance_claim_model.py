import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Optional
import io

# Configure page
st.set_page_config(
    page_title="Travel Insurance Claim Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained model pipeline with caching."""
    try:
        model = joblib.load('travel_insurance_claim_model.sav')
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'travel_insurance_claim_model.sav' not found. Please ensure the file is in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def get_feature_options():
    """Define the categorical feature options based on training data."""
    return {
        'Agency_type': ['Airlines', 'Travel Agency'],
        'Distribution_channel': ['Online', 'Offline'],
        'Agency': ['C2B', 'EPX', 'JZI', 'CWT', 'LWC', 'ART', 'CSR', 'RAB', 'KML', 'SSI', 'TST', 'TTW', 'ADM', 'CCR', 'CBH'],
        'Product_name': [
            'Annual Silver Plan', 'Cancellation Plan', 'Basic Plan', '2 way Comprehensive Plan',
            'Bronze Plan', '1 way Comprehensive Plan', 'Rental Vehicle Excess Insurance',
            'Single Trip Travel Protect Gold', 'Silver Plan', 'Value Plan', '24 Protect',
            'Annual Travel Protect', 'Single Trip Travel Protect Silver', 'Individual Comprehensive Plan',
            'Gold Plan', 'Annual Gold Plan', 'Child Comprehensive Plan', 'Premier Plan',
            'Annual Travel Protect Single Trip', 'Travel Protect Platinum', 'Annual Travel Protect Platinum',
            'Spouse or Parents Comprehensive Plan', 'Travel Cruise Protect Family'
        ],
        'Destination': [
            'SINGAPORE', 'MALAYSIA', 'INDIA', 'UNITED STATES', 'KOREA', 'REPUBLIC OF', 'THAILAND',
            'GERMANY', 'JAPAN', 'INDONESIA', 'VIET NAM', 'AUSTRALIA', 'FINLAND', 'UNITED KINGDOM',
            'SRI LANKA', 'SPAIN', 'HONG KONG', 'MACAO', 'CHINA', 'UNITED ARAB EMIRATES', 'IRAN',
            'ISLAMIC REPUBLIC OF', 'TAIWAN', 'PROVINCE OF CHINA', 'POLAND', 'CANADA', 'OMAN',
            'PHILIPPINES', 'GREECE', 'BELGIUM', 'TURKEY', 'BRUNEI DARUSSALAM', 'DENMARK',
            'SWITZERLAND', 'NETHERLANDS', 'SWEDEN', 'MYANMAR', 'KENYA', 'CZECH REPUBLIC', 'FRANCE',
            'RUSSIAN FEDERATION', 'PAKISTAN', 'ARGENTINA', 'TANZANIA', 'UNITED REPUBLIC OF',
            'SERBIA', 'ITALY', 'CROATIA', 'NEW ZEALAND', 'PERU', 'MONGOLIA', 'CAMBODIA', 'QATAR',
            'NORWAY', 'LUXEMBOURG', 'MALTA', 'LAO PEOPLE\'S DEMOCRATIC REPUBLIC', 'ISRAEL',
            'SAUDI ARABIA', 'AUSTRIA', 'PORTUGAL', 'NEPAL', 'UKRAINE', 'ESTONIA', 'ICELAND',
            'BRAZIL', 'MEXICO', 'CAYMAN ISLANDS', 'PANAMA', 'BANGLADESH', 'TURKMENISTAN',
            'BAHRAIN', 'KAZAKHSTAN', 'TUNISIA', 'IRELAND', 'ETHIOPIA', 'SOUTHERN MARIANA ISLANDS',
            'MALDIVES', 'SOUTH AFRICA', 'VENEZUELA', 'COSTA RICA', 'JORDAN', 'MALI', 'AZERBAIJAN',
            'HUNGARY', 'BHUTAN', 'BELARUS', 'ECUADOR', 'UZBEKISTAN', 'CHILE', 'FIJI',
            'PAPUA NEW GUINEA', 'ANGOLA', 'FRENCH POLYNESIA', 'NIGERIA', 'MACEDONIA',
            'THE FORMER YUGOSLAV REPUBLIC OF', 'NAMIBIA', 'GEORGIA', 'COLOMBIA', 'SLOVENIA',
            'EGYPT', 'ZIMBABWE', 'BULGARIA', 'BERMUDA', 'URUGUAY', 'GUINEA', 'GHANA', 'BOLIVIA',
            'PLURINATIONAL STATE OF', 'TRINIDAD AND TOBAGO', 'VANUATU', 'GUAM', 'UGANDA',
            'JAMAICA', 'LATVIA', 'ROMANIA', 'REPUBLIC OF MONTENEGRO', 'KYRGYZSTAN', 'GUATEMALA',
            'RWANDA', 'BOTSWANA', 'GUYANA', 'LITHUANIA', 'GUINEA-BISSAU', 'SENEGAL', 'CAMEROON',
            'SAMOA', 'PUERTO RICO', 'TAJIKISTAN', 'ARMENIA', 'FAROE ISLANDS', 'DOMINICAN REPUBLIC',
            'MOLDOVA', 'REPUBLIC OF BENIN', 'REUNION'
        ]
    }

def validate_numeric_input(value: float, field_name: str, min_val: float = None, max_val: float = None) -> bool:
    """Validate numeric input ranges."""
    if pd.isna(value):
        st.error(f"❌ {field_name} cannot be empty")
        return False
    
    if min_val is not None and value < min_val:
        st.error(f"❌ {field_name} must be >= {min_val}")
        return False
        
    if max_val is not None and value > max_val:
        st.error(f"❌ {field_name} must be <= {max_val}")
        return False
    
    return True

def create_input_dataframe(duration: float, net_sales: float, commission: float, age: float,
                          agency_type: str, distribution_channel: str, agency: str,
                          product_name: str, destination: str) -> pd.DataFrame:
    """Create a DataFrame with the input features in the correct format."""
    return pd.DataFrame({
        'Duration': [duration],
        'Net_sales': [net_sales],
        'Commision': [commission],  # Note: keeping original column name from training
        'Age': [age],
        'Agency_type': [agency_type],
        'Distribution_channel': [distribution_channel],
        'Agency': [agency],
        'Product_name': [product_name],
        'Destination': [destination]
    })

def make_prediction(model, input_df: pd.DataFrame, threshold: float = 0.5):
    """Make prediction using the loaded model."""
    try:
        # Get probability predictions
        proba = model.predict_proba(input_df)[:, 1]  # Probability of class 1
        
        # Apply threshold to get binary predictions
        pred_labels = (proba >= threshold).astype(int)
        
        return proba, pred_labels
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None, None

def display_model_coefficients(model):
    """Display top feature coefficients for model transparency."""
    try:
        # Get feature names from preprocessing step
        if hasattr(model, 'named_steps'):
            preprocessor = model.named_steps.get('preprocessing')
            classifier = model.named_steps.get('model')
            
            if preprocessor and classifier and hasattr(classifier, 'coef_'):
                feature_names = preprocessor.get_feature_names_out()
                coefficients = classifier.coef_[0]
                
                # Create coefficient DataFrame
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients,
                    'Abs_Coefficient': np.abs(coefficients)
                }).sort_values('Abs_Coefficient', ascending=False).head(10)
                
                st.subheader("🔍 Top 10 Model Features (by coefficient magnitude)")
                st.dataframe(
                    coef_df[['Feature', 'Coefficient']],
                    use_container_width=True,
                    hide_index=True
                )
                
                with st.expander("📊 Coefficient Interpretation"):
                    st.write("""
                    - **Positive coefficients** increase the probability of submitting a claim
                    - **Negative coefficients** decrease the probability of submitting a claim  
                    - **Larger absolute values** indicate stronger influence on the prediction
                    """)
    except Exception as e:
        st.warning(f"Could not display model coefficients: {str(e)}")

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("✈️ Travel Insurance Claim Predictor")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    # Sidebar for model information
    with st.sidebar:
        st.header("📊 Model Information")
        st.info("**Model Type:** Logistic Regression Pipeline")
        st.info("**Artifact:** travel_insurance_claim_model.sav")
        st.info("**Preprocessing:** Included in pipeline")
        
        # Threshold control
        st.header("⚙️ Prediction Settings")
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Probability threshold for classifying as 'likely to claim'"
        )
        
        st.markdown(f"**Current threshold:** {threshold}")
        st.markdown("- Prob ≥ {:.2f} → Likely to claim (1)".format(threshold))
        st.markdown("- Prob < {:.2f} → Unlikely to claim (0)".format(threshold))
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction", "🔍 Model Insights"])
    
    with tab1:
        st.header("Single Customer Prediction")
        
        # Get feature options
        feature_options = get_feature_options()
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Numerical Features")
                duration = st.number_input(
                    "Duration (days)",
                    min_value=1.0,
                    max_value=740.0,
                    value=25.0,
                    step=1.0,
                    help="Trip duration in days (1-740)"
                )
                
                net_sales = st.number_input(
                    "Net Sales",
                    min_value=-357.5,
                    max_value=682.0,
                    value=28.0,
                    step=0.1,
                    help="Net sales amount (-357.5 to 682.0)"
                )
                
                commission = st.number_input(
                    "Commission",
                    min_value=0.0,
                    max_value=262.76,
                    value=0.0,
                    step=0.01,
                    help="Commission amount (0.0 to 262.76)"
                )
                
                age = st.number_input(
                    "Customer Age",
                    min_value=0.0,
                    max_value=88.0,
                    value=36.0,
                    step=1.0,
                    help="Customer age (0-88 years)"
                )
            
            with col2:
                st.subheader("📝 Categorical Features")
                agency_type = st.selectbox(
                    "Agency Type",
                    options=feature_options['Agency_type'],
                    help="Type of travel agency"
                )
                
                distribution_channel = st.selectbox(
                    "Distribution Channel",
                    options=feature_options['Distribution_channel'],
                    help="Sales channel (Online/Offline)"
                )
                
                agency = st.selectbox(
                    "Agency Code",
                    options=feature_options['Agency'],
                    help="Specific agency identifier"
                )
                
                product_name = st.selectbox(
                    "Insurance Product",
                    options=feature_options['Product_name'],
                    help="Type of insurance product purchased"
                )
                
                destination = st.selectbox(
                    "Destination Country",
                    options=feature_options['Destination'],
                    help="Travel destination country"
                )
            
            # Submit button
            submitted = st.form_submit_button("🎯 Predict Claim Probability", use_container_width=True)
        
        if submitted:
            # Validate inputs
            valid = True
            valid &= validate_numeric_input(duration, "Duration", 1.0, 740.0)
            valid &= validate_numeric_input(net_sales, "Net Sales", -357.5, 682.0)
            valid &= validate_numeric_input(commission, "Commission", 0.0, 262.76)
            valid &= validate_numeric_input(age, "Age", 0.0, 88.0)
            
            if valid:
                # Create input DataFrame
                input_df = create_input_dataframe(
                    duration, net_sales, commission, age,
                    agency_type, distribution_channel, agency,
                    product_name, destination
                )
                
                # Make prediction
                proba, pred_label = make_prediction(model, input_df, threshold)
                
                if proba is not None:
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Claim Probability",
                            f"{proba[0]:.3f}",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Predicted Label",
                            pred_label[0],
                            delta=None
                        )
                    
                    with col3:
                        confidence = "High" if abs(proba[0] - 0.5) > 0.3 else "Medium" if abs(proba[0] - 0.5) > 0.15 else "Low"
                        st.metric(
                            "Confidence",
                            confidence,
                            delta=None
                        )
                    
                    # Interpretation
                    if pred_label[0] == 1:
                        st.success(f"🔴 **Prediction: Likely to submit a claim** (probability: {proba[0]:.1%})")
                        st.info("💡 This customer has a higher risk profile for submitting an insurance claim.")
                    else:
                        st.success(f"🟢 **Prediction: Unlikely to submit a claim** (probability: {proba[0]:.1%})")
                        st.info("💡 This customer has a lower risk profile for submitting an insurance claim.")
    
    with tab2:
        st.header("Batch Prediction from CSV")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch predictions",
            type=['csv'],
            help="CSV must contain columns: Duration, Net_sales, Commision, Age, Agency_type, Distribution_channel, Agency, Product_name, Destination"
        )
        
        if uploaded_file is not None:
            try:
                # Load CSV
                batch_df = pd.read_csv(uploaded_file)
                
                st.subheader("📄 Uploaded Data Preview")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                # Validate required columns
                required_columns = ['Duration', 'Net_sales', 'Commision', 'Age', 'Agency_type', 
                                  'Distribution_channel', 'Agency', 'Product_name', 'Destination']
                missing_columns = [col for col in required_columns if col not in batch_df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {missing_columns}")
                else:
                    # Make batch predictions
                    with st.spinner("Making predictions..."):
                        probas, pred_labels = make_prediction(model, batch_df, threshold)
                    
                    if probas is not None:
                        # Add predictions to DataFrame
                        results_df = batch_df.copy()
                        results_df['proba_class_1'] = probas
                        results_df['pred_label'] = pred_labels
                        
                        st.subheader("📊 Prediction Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records", len(results_df))
                        with col2:
                            st.metric("Predicted Claims", int(pred_labels.sum()))
                        with col3:
                            st.metric("Claim Rate", f"{pred_labels.mean():.1%}")
                        
                        # Download button
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv_buffer.getvalue(),
                            file_name="prediction_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with tab3:
        st.header("Model Insights & Transparency")
        
        # Display model coefficients
        display_model_coefficients(model)
        
        # Additional model information
        st.subheader("📋 Model Details")
        
        model_info = f"""
        **Model Architecture:** Scikit-learn Pipeline
        - **Preprocessing:** Automated encoding and scaling (built into pipeline)
        - **Algorithm:** Logistic Regression
        - **Target Variable:** Binary (0 = No claim, 1 = Claim)
        - **Current Threshold:** {threshold}
        
        **Feature Summary:**
        - **Numerical Features:** Duration, Net_sales, Commission, Age
        - **Categorical Features:** Agency_type, Distribution_channel, Agency, Product_name, Destination
        - **Total Training Features:** Varies based on categorical encoding
        """
        
        st.markdown(model_info)
        
        st.subheader("⚠️ Important Notes")
        st.warning("""
        - This model makes predictions based on historical patterns
        - Results should be used as decision support, not definitive predictions
        - Consider business context and domain expertise when interpreting results
        - Monitor model performance regularly and retrain as needed
        """)

if __name__ == "__main__":
    main()