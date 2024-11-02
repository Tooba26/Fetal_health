import streamlit as st
import pandas as pd
import joblib
from PIL import Image

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffafbd, #ffc3a0);
        font-family: 'Arial', sans-serif;      
    }
            
    .flex-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        padding: 10px;
    }
    .title {
        color: #3A3B3C;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .header {
        font-size: 1.2rem;
        color: #333333;
        margin-bottom: 20px;
    }
    .input-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .btn-primary {
        background-color: #F0A8D0 !important;
        color: #fff !important;
        border-radius: 8px !important;
        padding: 10px 15px !important;
    }
    .btn-primary:hover {
        background-color: #F0A8D0 !important;
    }
    .centered-image {
        width: 100px; /* Adjust width as needed */
    }
            
    .st-emotion-cache-yk7at5 {
        background-color: #F7B5CA !important;
        border: 1px solid #F7B5CA
            
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

image = Image.open("fh.png")


# Create side-by-side layout for the image and title
col1, col2 = st.columns([1, 5])  # Adjust ratios as needed
with col1:
    st.image(image, width=100)  # Adjust width as needed

with col2:
    st.markdown('<h1 class="title">Fetal Health Prediction</h1>', unsafe_allow_html=True)


# Input fields for user to enter data
st.header("Input Instructions")
# Input instructions in collapsible expander section
with st.expander("📋 Input Instructions (Click to Expand)", expanded=False):
    st.markdown("""
        1. **Baseline Value**: Enter the baseline fetal heart rate in beats per minute (BPM) (Range: 106-160).
        2. **Accelerations**: Enter the number of accelerations detected during monitoring (Range: 0.0-0.019).
        3. **Fetal Movement**: Enter the number of fetal movements observed (Range: 0.0-0.481).
        4. **Uterine Contractions**: Enter the number of uterine contractions detected (Range: 0.0-0.015).
        5. **Light Decelerations**: Enter the number of light decelerations recorded (Range: 0.0-0.015).
        6. **Severe Decelerations**: Enter the number of severe decelerations observed (Range: 0.0-0.001).
        7. **Prolonged Decelerations**: Enter the number of prolonged decelerations detected (Range: 0.0-0.005).
        8. **Abnormal Short Term Variability**: Value for abnormal short-term variability (Range: 12-87).
        9. **Mean Value of Short Term Variability**: Mean value of short-term variability (Range: 0.2-7.0).
        10. **Percentage of Time with Abnormal Long Term Variability**: (Range: 0.0-91.0).
        11. **Histogram Width**: Enter the histogram width value (Range: 0.0-40.0).
        12. **Histogram Min**: Minimum value in the histogram (Range: 50-159).
        13. **Histogram Max**: Maximum value in the histogram (Range: 122-238).
        14. **Histogram Number of Peaks**: Number of peaks detected in the histogram (Range: 0-18).
        15. **Histogram Number of Zeroes**: Number of zero values in the histogram (Range: 0-10).
        16. **Histogram Mode**: Mode value from the histogram (Range: 60-187).
        17. **Histogram Mean**: Mean value from the histogram (Range: 73-182).
        18. **Histogram Median**: Median value from the histogram (Range: 77-186).
        19. **Histogram Variance**: Variance value from the histogram (Range: 0.0-269.0).
        20. **Histogram Tendency**: Tendency value of the histogram (Range: -1.0 to 1.0).
        21. **Mean Value of Long Term Variability**: Mean value of long-term variability (Range: -1.0 to 1.0).
    """)

st.markdown('<p class="header">Enter the following parameters for prediction:</p>', unsafe_allow_html=True)
with st.container():
    # Collect user input
    baseline_value = st.number_input("Baseline Value", min_value=106, max_value=160)
    accelerations = st.number_input("Accelerations", min_value=0.0, max_value=0.019)
    fetal_movement = st.number_input("Fetal Movement", min_value=0.0, max_value=0.481)
    uterine_contractions = st.number_input("Uterine Contractions", min_value=0.0, max_value=0.015)
    light_decelerations = st.number_input("Light Decelerations", min_value=0.0, max_value=0.015)
    severe_decelerations = st.number_input("Severe Decelerations", min_value=0.0, max_value=0.001)
    prolonged_decelerations = st.number_input("Prolonged Decelerations", min_value=0.0, max_value=0.005)
    abnormal_short_term_variability = st.number_input("Abnormal Short Term Variability", min_value=12, max_value=87)
    mean_value_of_short_term_variability = st.number_input("Mean Value of Short Term Variability", min_value=0.2, max_value=7.0)
    percentage_of_time_with_abnormal_long_term_variability = st.number_input("Percentage of Time with Abnormal Long Term Variability", min_value=0.0, max_value=91.0)
    histogram_width = st.number_input("Histogram Width", min_value=0.0, max_value=40.0)
    histogram_min = st.number_input("Histogram Min", min_value=50, max_value=159)
    histogram_max = st.number_input("Histogram Max", min_value=122, max_value=238)
    histogram_number_of_peaks = st.number_input("Histogram Number of Peaks", min_value=0, max_value=18)
    histogram_number_of_zeroes = st.number_input("Histogram Number of Zeroes", min_value=0, max_value=10)
    histogram_mode = st.number_input("Histogram Mode", min_value=60, max_value=187)
    histogram_mean = st.number_input("Histogram Mean", min_value=73, max_value=182)
    histogram_median = st.number_input("Histogram Median", min_value=77, max_value=186)
    histogram_variance = st.number_input("Histogram Variance", min_value=0.0, max_value=269.0)
    histogram_tendency = st.number_input("Histogram Tendency", min_value=-1.0, max_value=1.0)
    mean_value_of_long_term_variability = st.number_input("mean_value_of_long_term_variability", min_value=-1.0, max_value=1.0)


st.markdown("""
                **Prediction Status:**
                1. Normal
                2. Suspect
                3. Pathalogical
                """)
# When the user clicks the button, make a prediction
if st.button("Predict", key="predict_button", type="primary"):
    # Prepare the input data for the model
    input_data = pd.DataFrame([[baseline_value, accelerations, fetal_movement, uterine_contractions,
                                 light_decelerations, severe_decelerations, prolonged_decelerations,
                                 abnormal_short_term_variability, mean_value_of_short_term_variability,
                                 percentage_of_time_with_abnormal_long_term_variability, histogram_width,
                                 histogram_min, histogram_max, histogram_number_of_peaks,
                                 histogram_number_of_zeroes, histogram_mode, histogram_mean,
                                 histogram_median, histogram_variance, histogram_tendency, mean_value_of_long_term_variability]],
                               columns=["baseline value", "accelerations", "fetal_movement", "uterine_contractions",
                                        "light_decelerations", "severe_decelerations", "prolongued_decelerations",
                                        "abnormal_short_term_variability", "mean_value_of_short_term_variability",
                                        "percentage_of_time_with_abnormal_long_term_variability", "histogram_width",
                                        "histogram_min", "histogram_max", "histogram_number_of_peaks",
                                        "histogram_number_of_zeroes", "histogram_mode", "histogram_mean",
                                        "histogram_median", "histogram_variance", "histogram_tendency", "mean_value_of_long_term_variability"])
    

    # Make prediction
    prediction = model.predict(input_data)

    # Display the result
    st.write("Predicted Fetal Health Status:", int(prediction[0]))

    
