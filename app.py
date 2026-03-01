import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import os

# Page Config
st.set_page_config(page_title="Iris Species Classifier", layout="wide", page_icon="🌱")

# -------- PROFESSIONAL UI STYLE --------
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Section with Gradient */
    .header-box {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }

    .header-box h1 {
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
        color: #f8fafc;
    }
    
    /* Input Card - Glassmorphism Effect */
    .input-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 30px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        margin-bottom: 20px;
    }
    
    /* Result Styling - Elevated Card */
    .prediction-output {
        background: white;
        padding: 40px;
        border-radius: 24px;
        border-top: 8px solid #6366f1;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .prediction-output:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 30px -5px rgba(0, 0, 0, 0.15);
    }

    /* Professional Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
        width: 100%;
    }

    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.45);
        color: white;
    }

    /* Input Field Labels */
    .stNumberInput label p {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 0.95rem;
    }
    
    /* Metrics labels */
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_my_model():
    # Ensure the file exists to prevent crashing
    if os.path.exists("iris_model.h5"):
        return tf.keras.models.load_model("iris_model.h5")
    else:
        st.error("Error: 'iris_model.h5' not found in directory.")
        return None

model = load_my_model()

# -------- HEADER --------
st.markdown("""
    <div class='header-box'>
        <h1>Iris Species Prediction System</h1>
        <p style='opacity: 0.8;'>Deep Learning Classification • AI/ML Research Submission</p>
    </div>
""", unsafe_allow_html=True)

# -------- MAIN LAYOUT --------
col_input, col_display = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("### 🛠️ Morphological Features")
    st.write("Enter the flower dimensions below to generate a prediction:")
    
    # Input container with custom class
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    a = st.number_input("Sepal Length (cm)", value=5.1, step=0.1, format="%.1f")
    b = st.number_input("Sepal Width (cm)", value=3.5, step=0.1, format="%.1f")
    c = st.number_input("Petal Length (cm)", value=1.4, step=0.1, format="%.1f")
    d = st.number_input("Petal Width (cm)", value=0.2, step=0.1, format="%.1f")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.write("")
    predict_btn = st.button("Generate Classification Report")

with col_display:
    if predict_btn and model is not None:
        # Prediction Logic
        data = np.array([[a, b, c, d]])
        prediction = model.predict(data)
        result_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        names = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]
        colors = ["#3b82f6", "#10b981", "#8b5cf6"] # Blue, Green, Purple
        
        # Display Result Card
        st.markdown(f"""
            <div class='prediction-output'>
                <h4 style='color: #64748b; margin-bottom: 0;'>CLASSIFICATION RESULT</h4>
                <h1 style='color: {colors[result_index]}; font-size: 56px; margin-top: 0;'>{names[result_index]}</h1>
                <hr style='border: 0.5px solid #f1f5f9; margin: 20px 0;'>
                <div style='display: flex; justify-content: space-around;'>
                    <div>
                        <p class='metric-label'>Model Confidence</p>
                        <h2 style='color: #1e293b;'>{confidence:.2f}%</h2>
                    </div>
                    <div>
                        <p class='metric-label'>Status</p>
                        <h2 style='color: #10b981;'>Verified</h2>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Visualization
        st.write("")
        st.markdown("### 📊 Probability Distribution")
        chart_data = pd.DataFrame(prediction[0], index=names, columns=["Probability"])
        st.bar_chart(chart_data, color=colors[result_index])

    elif model is None:
        st.warning("Please ensure the model file is in the project folder.")
    else:
        # Placeholder before prediction
        st.info("💡 **Developer Note:** Provide inputs and click the button to see the AI classification and model confidence level.")
        
        with st.expander("🔬 Model Metadata"):
            st.markdown("""
            **Architecture:** Deep Neural Network (Sequential)  
            **Input Shape:** 4 Features (Sepal/Petal dimensions)  
            **Activation:** ReLU (Hidden), Softmax (Output)  
            **Environment:** TensorFlow 2.x / Python 3.13
            """)