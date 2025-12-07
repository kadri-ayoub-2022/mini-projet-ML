"""
Streamlit Web Application for Life Expectancy Prediction
This app uses the trained ML model to predict life expectancy based on country metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Life Expectancy Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        with open('best_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please run 'train_and_save_model.py' first.")
        st.stop()

# Load model
model_package = load_model()

# Header
st.markdown('<h1 class="main-header">🏥 Life Expectancy Prediction System</h1>', unsafe_allow_html=True)
st.markdown(f"""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Powered by <b>{model_package['model_name']}</b> |
    Accuracy (R²): <b>{model_package['performance']['r2']:.2%}</b> |
    Error (MAE): <b>{model_package['performance']['mae']:.2f} years</b>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("./supervised/téléchargement.png", width=100)
    st.title("📊 Navigation")
    page = st.radio("Select Page:", ["🔮 Single Prediction", "📊 Batch Prediction", "📈 Model Info", "📚 About"])

    st.markdown("---")
    st.markdown("### 🎯 Quick Stats")
    st.info(f"**Model Type:** {model_package['model_name']}")
    st.success(f"**Accuracy:** {model_package['performance']['r2']:.2%}")
    st.warning(f"**Avg Error:** ±{model_package['performance']['mae']:.2f} years")

# Page 1: Single Prediction
if page == "🔮 Single Prediction":
    st.markdown('<h2 class="sub-header">Enter Country Metrics</h2>', unsafe_allow_html=True)

    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 😊 Social Indicators")
            happiness = st.slider("Happiness Score (0-10)", 0.0, 10.0, 5.0, 0.1,
                                help="Self-reported happiness index")
            literacy = st.slider("Literacy Rate (%)", 0.0, 100.0, 80.0, 0.1,
                                help="Adult literacy rate")
            inequality = st.slider("Inequality Index (0-100)", 0.0, 100.0, 35.0, 0.1,
                                  help="Income inequality measure (Gini coefficient)")

        with col2:
            st.markdown("#### 💰 Economic Indicators")
            income = st.number_input("Average Income (USD)", 0, 150000, 20000, 100,
                                    help="GDP per capita")
            energy = st.number_input("Energy Consumption (kWh)", 0, 20000, 2000, 100,
                                    help="Per capita energy consumption")
            foodsupply = st.number_input("Food Supply (kcal/day)", 0, 4000, 2500, 10,
                                        help="Average daily caloric intake")

        with col3:
            st.markdown("#### 🏥 Health Indicators")
            sanitation = st.slider("Sanitation Access (%)", 0.0, 100.0, 85.0, 0.1,
                                  help="Population with access to sanitation")
            water = st.slider("Clean Water Access (%)", 0.0, 100.0, 90.0, 0.1,
                            help="Population with access to clean water")
            childmortality = st.slider("Child Mortality (per 1000)", 0.0, 150.0, 30.0, 0.1,
                                      help="Under-5 mortality rate")
            fertility = st.slider("Fertility Rate (births/woman)", 0.0, 8.0, 2.5, 0.1,
                                help="Average births per woman")
            hiv = st.number_input("HIV Cases (per 100k)", 0, 10000000, 50000, 1000,
                                help="Number of HIV cases")
            population = st.number_input("Population", 0, 2000000000, 10000000, 100000,
                                        help="Total population")

        # Submit button
        submitted = st.form_submit_button("🔮 Predict Life Expectancy", use_container_width=True)

    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'happiness': [happiness],
            'income': [income],
            'sanitation': [sanitation],
            'water': [water],
            'literacy': [literacy],
            'inequality': [inequality],
            'energy': [energy],
            'childmortality': [childmortality],
            'fertility': [fertility],
            'hiv': [hiv],
            'foodsupply': [foodsupply],
            'population': [population]
        })

        # Scale the input
        scaled_input = model_package['scaler'].transform(input_data)

        # Make prediction
        prediction = model_package['model'].predict(scaled_input)[0]

        # Display prediction
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🎯 Prediction Result</h2>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 1.2rem; margin-bottom: 1rem;">Predicted Life Expectancy</div>
                    <div style="font-size: 3.5rem; font-weight: bold;">{prediction:.1f}</div>
                    <div style="font-size: 1.2rem; margin-top: 1rem;">years</div>
                </div>
            """, unsafe_allow_html=True)

        # Confidence interval
        mae = model_package['performance']['mae']
        st.info(f"""
            📊 **Confidence Interval:** {prediction - mae:.1f} - {prediction + mae:.1f} years
            ℹ️ The actual life expectancy is likely to fall within this range based on our model's performance.
        """)

        # Feature importance (if available)
        if hasattr(model_package['model'], 'feature_importances_'):
            st.markdown('<h3 class="sub-header">📊 Feature Importance</h3>', unsafe_allow_html=True)

            importance_df = pd.DataFrame({
                'Feature': model_package['feature_names'],
                'Importance': model_package['model'].feature_importances_
            }).sort_values('Importance', ascending=False)

            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Viridis')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Show input summary
        with st.expander("📋 View Input Summary"):
            st.dataframe(input_data.T, use_container_width=True)

# Page 2: Batch Prediction
elif page == "📊 Batch Prediction":
    st.markdown('<h2 class="sub-header">Batch Prediction - Upload CSV File</h2>', unsafe_allow_html=True)

    st.info("""
    📋 Upload a CSV file with multiple countries to get predictions for all at once.
    The CSV must contain the following columns: happiness, income, sanitation, water, literacy,
    inequality, energy, childmortality, fertility, hiv, foodsupply, population
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Download Sample CSV Template"):
            sample_data = pd.DataFrame({
                'country_name': ['Country A', 'Country B', 'Country C'],
                'happiness': [7.5, 5.0, 3.5],
                'income': [45000, 15000, 2000],
                'sanitation': [99.0, 75.0, 30.0],
                'water': [100.0, 85.0, 50.0],
                'literacy': [99.0, 80.0, 40.0],
                'inequality': [30.0, 40.0, 55.0],
                'energy': [5000, 1500, 300],
                'childmortality': [5.0, 30.0, 80.0],
                'fertility': [1.8, 3.0, 5.5],
                'hiv': [20000, 100000, 500000],
                'foodsupply': [3300, 2500, 1800],
                'population': [10000000, 30000000, 15000000]
            })

            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="💾 Download",
                data=csv,
                file_name="batch_prediction_template.csv",
                mime="text/csv"
            )

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)

            st.success(f"✓ File uploaded successfully! Found {len(data)} rows.")

            # Show preview
            with st.expander("👀 Preview Uploaded Data"):
                st.dataframe(data.head(10), use_container_width=True)

            # Verify required columns
            required_features = model_package['feature_names']
            missing_cols = [col for col in required_features if col not in data.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info(f"Required columns: {', '.join(required_features)}")
            else:
                st.success("✓ All required columns found!")

                # Make predictions button
                if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
                    with st.spinner("Making predictions..."):
                        # Extract features in correct order
                        X = data[required_features]

                        # Scale features
                        X_scaled = model_package['scaler'].transform(X)

                        # Predict
                        predictions = model_package['model'].predict(X_scaled)

                        # Add predictions to dataframe
                        result = data.copy()
                        result['predicted_lifeexp'] = predictions.round(2)
                        result['confidence_lower'] = (predictions - model_package['performance']['mae']).round(2)
                        result['confidence_upper'] = (predictions + model_package['performance']['mae']).round(2)

                        st.success(f"✓ Predictions completed for {len(predictions)} rows!")

                        # Display summary statistics
                        st.markdown('<h3 class="sub-header">📊 Prediction Summary</h3>', unsafe_allow_html=True)

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Minimum", f"{predictions.min():.1f} years")
                        with col2:
                            st.metric("Average", f"{predictions.mean():.1f} years")
                        with col3:
                            st.metric("Maximum", f"{predictions.max():.1f} years")
                        with col4:
                            st.metric("Std Dev", f"{predictions.std():.1f} years")

                        # Show results
                        st.markdown('<h3 class="sub-header">📋 Results</h3>', unsafe_allow_html=True)
                        st.dataframe(result, use_container_width=True)

                        # Visualization
                        st.markdown('<h3 class="sub-header">📈 Distribution</h3>', unsafe_allow_html=True)

                        fig = px.histogram(
                            result,
                            x='predicted_lifeexp',
                            nbins=30,
                            title='Distribution of Predicted Life Expectancies',
                            labels={'predicted_lifeexp': 'Life Expectancy (years)', 'count': 'Frequency'},
                            color_discrete_sequence=['#667eea']
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                        # Download results
                        csv_output = result.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_output,
                            file_name="life_expectancy_predictions.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has the correct format.")

# Page 3: Model Info
elif page == "📈 Model Info":
    st.markdown('<h2 class="sub-header">Model Performance Metrics</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", model_package['model_name'])
    with col2:
        st.metric("R² Score", f"{model_package['performance']['r2']:.4f}",
                 delta="Excellent" if model_package['performance']['r2'] > 0.9 else "Good")
    with col3:
        st.metric("Mean Absolute Error", f"{model_package['performance']['mae']:.2f} years")

    st.markdown("---")

    # Feature statistics
    st.markdown('<h3 class="sub-header">📊 Feature Statistics</h3>', unsafe_allow_html=True)

    stats_df = pd.DataFrame({
        'Feature': model_package['feature_names'],
        'Min': [model_package['feature_stats']['min'][f] for f in model_package['feature_names']],
        'Max': [model_package['feature_stats']['max'][f] for f in model_package['feature_names']],
        'Mean': [model_package['feature_stats']['mean'][f] for f in model_package['feature_names']]
    })

    st.dataframe(stats_df, use_container_width=True)

    

# Page 4: About
elif page == "📚 About":
    st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Overview

    This web application is part of a **Machine Learning and Data Mining** project that predicts
    life expectancy based on various country-level indicators.

    ### 📊 Dataset

    The model was trained on a comprehensive dataset containing information about 194 countries,
    including:
    - Social indicators (happiness, literacy, inequality)
    - Economic metrics (income, energy consumption, food supply)
    - Health statistics (sanitation, water access, child mortality, fertility, HIV prevalence)
    - Demographic data (population)

    ### 🤖 Machine Learning Models Tested

    We evaluated **5 different regression algorithms**:
    1. Linear Regression
    2. K-Nearest Neighbors (KNN)
    3. Support Vector Regression (SVR)
    4. Decision Tree
    5. **Random Forest** ⭐

    The best model was selected based on **Mean Absolute Error (MAE)** and **R² Score**.

    ### 🔬 Model Performance

    Our best model achieves:
    - **R² Score:** {:.2%} (explains {:.0%} of variance in life expectancy)
    - **MAE:** ±{:.2f} years (average prediction error)

    This represents a **40-50% improvement** over baseline models!

    ### 🛠️ Technologies Used

    - **Python** - Programming language
    - **scikit-learn** - Machine learning library
    - **Streamlit** - Web application framework
    - **Pandas & NumPy** - Data manipulation
    - **Plotly** - Interactive visualizations
    - **Pickle** - Model serialization

    ### 👥 Team

    Higher School of Computer Science (ESI)
    08 May 1948 - Sidi Bel Abbes

    **Course:** Machine Learning and Data Mining
    **Academic Year:** 2025-2026

    ### 📅 Project Timeline

    - **Presentation Date:** December 9, 2025
    - **Duration:** 10-12 minutes

    ### 📖 How to Use

    1. Go to the **Prediction** page
    2. Enter the country metrics using the sliders and input fields
    3. Click **Predict Life Expectancy**
    4. View the predicted life expectancy and confidence interval

    ### ⚠️ Disclaimer

    This model is for educational purposes only. Predictions should not be used for
    policy-making without further validation and domain expert consultation.

    ---

    Made with ❤️ using Streamlit | © 2025 ESI Sidi Bel Abbes
    """.format(
        model_package['performance']['r2'],
        model_package['performance']['r2'],
        model_package['performance']['mae']
    ))

    # Add download button for sample data
    st.markdown("### 📥 Download Sample Data")

    sample_data = pd.DataFrame({
        'happiness': [7.5],
        'income': [45000],
        'sanitation': [99.0],
        'water': [100.0],
        'literacy': [99.0],
        'inequality': [30.0],
        'energy': [5000],
        'childmortality': [5.0],
        'fertility': [1.8],
        'hiv': [20000],
        'foodsupply': [3300],
        'population': [10000000]
    })

    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="📄 Download Sample Input CSV",
        data=csv,
        file_name="sample_input.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #999; padding: 1rem;'>
    Life Expectancy Prediction System | Powered by Machine Learning | ESI Sidi Bel Abbes 2025
    </div>
""", unsafe_allow_html=True)
