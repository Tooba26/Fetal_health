import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Streamlit app title
st.title("Fetal Health Prediction")

# Input fields for user to enter data
st.header("Input Instructions")
st.markdown("""
1. **Baseline Value**: (Min: 106, Max: 160) - Enter the baseline fetal heart rate in beats per minute (BPM).
2. **Accelerations**: (Min: 0.0, Max: 0.019) - Enter the number of accelerations detected during monitoring.
3. **Fetal Movement**: (Min: 0.0, Max: 0.481) - Enter the number of fetal movements observed.
4. **Uterine Contractions**: (Min: 0.0, Max: 0.015) - Enter the number of uterine contractions detected.
5. **Light Decelerations**: (Min: 0.0, Max: 0.015) - Enter the number of light decelerations recorded.
6. **Severe Decelerations**: (Min: 0.0, Max: 0.001) - Enter the number of severe decelerations observed.
7. **Prolonged Decelerations**: (Min: 0.0, Max: 0.005) - Enter the number of prolonged decelerations detected.
8. **Abnormal Short Term Variability**: (Min: 12, Max: 87) - Enter the value for abnormal short-term variability.
8. **Abnormal Short Term Variability**: (Min: 12, Max: 87) - Enter the value for abnormal short-term variability.
9. **Mean Value of Short Term Variability**: (Min: 0.2, Max: 7.0) - Enter the mean value of short-term variability.
10. **Percentage of Time with Abnormal Long Term Variability**: (Min: 0.0, Max: 91.0) - Enter the percentage of time with abnormal long-term variability.
11. **Histogram Width**: (Min: 0.0, Max: 40.0) - Enter the histogram width value.
12. **Histogram Min**: (Min: 50, Max: 159) - Enter the minimum value in the histogram.
13. **Histogram Max**: (Min: 122, Max: 238) - Enter the maximum value in the histogram.
14. **Histogram Number of Peaks**: (Min: 0, Max: 18) - Enter the number of peaks detected in the histogram.
15. **Histogram Number of Zeroes**: (Min: 0, Max: 10) - Enter the number of zero values in the histogram.
16. **Histogram Mode**: (Min: 60, Max: 187) - Enter the mode value from the histogram.
17. **Histogram Mean**: (Min: 73, Max: 182) - Enter the mean value from the histogram.
18. **Histogram Median**: (Min: 77, Max: 186) - Enter the median value from the histogram.
19. **Histogram Variance**: (Min: 0.0, Max: 269.0) - Enter the variance value from the histogram.
20. **Histogram Tendency**: (Min: -1.0, Max: 1.0) - Enter the tendency value of the histogram.
21. **mean_value_of_long_term_variability**: (Min: -1.0, Max: 1.0) - Enter the tendency value of the histogram.
""")

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
if st.button("Predict"):
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

    
