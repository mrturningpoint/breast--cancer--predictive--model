Breast Cancer Prediction

This project involves two main steps: data cleaning and model training/testing. The goal is to build a predictive model for breast cancer diagnosis using logistic regression. This model has been found to be highly accurate compared to other models.

Steps
1. Data Cleaning
The data cleaning process involves:

Reading the dataset from a CSV file.
Dropping the 'id' column as it is not needed for the analysis.
Mapping the 'diagnosis' column to numerical values (0 for benign, 1 for malignant).

2. Model Training and Testing
The model training and testing process includes:

Splitting the dataset into training and testing sets.
Scaling the feature values for better performance.
Training a logistic regression model on the training set.
Testing the model on the testing set and printing the accuracy and classification report.

Best Model

The logistic regression model used in this project has been found to be highly accurate in predicting breast cancer, outperforming other models in terms of accuracy.

Code Explanation

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_model(data):
    x = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scalar = StandardScaler()
    x_tr = scalar.fit_transform(x_train)
    x_te = scalar.transform(x_test)
    classifier = LogisticRegression()
    classifier.fit(x_tr, y_train)
    y_predict = classifier.predict(x_te)
    print("Accuracy:", accuracy_score(y_test, y_predict))
    print("Classification Report:\n", classification_report(y_test, y_predict))
    return scalar, classifier

def clean_data():
    data = pd.read_csv("data/breast-cancer.csv")
    data = data.drop('id', axis=1)
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
    return data

def main():
    data = clean_data()
    scalar, model = create_model(data)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scalar.pkl', 'wb') as f:
        pickle.dump(scalar, f)

if __name__ == '__main__':
    main()
File Structure

data/breast-cancer.csv: The dataset file.
model/model.pkl: The trained logistic regression model.
model/scalar.pkl: The standard scaler used for data transformation.

Running the Project

To run the project, simply execute the main function. This will clean the data, train the model, and save the trained model and scaler.

Streamlit Web App

This project includes a Streamlit web app for interactive breast cancer prediction. The app allows users to input cell measurements and visualize the data, as well as predict the probability of breast cancer diagnosis.

Code Explanation

import streamlit as st
import numpy as np
import pandas as pd
import pickle as pickl
import plotly.graph_objects as go

def clean_data():
    data = pd.read_csv("data/breast-cancer.csv")
    data = data.drop('id', axis=1)
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
    return data

def get_scaled_values(input_dict):
    data = clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict

def get_radar(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    return fig

def add_sidebar():
    st.sidebar.header("Cell Measurements")
    data = clean_data()
    input_dict = {}
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def add_prediction(input_data):
    model = pickl.load(open("model/model.pkl", "rb"))
    scalar = pickl.load(open("model/scalar.pkl", "rb"))
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    scaled_array = scalar.transform(input_array)
    prediction = model.predict(scaled_array)

    if prediction[0] == 0:
        st.write("Diagnosis: Benign")
    else:
        st.write("Diagnosis: Malignant")
    
    st.write("Probability of being benign: ", model.predict_proba(scaled_array)[0][0])
    st.write("Probability of being malignant: ", model.predict_proba(scaled_array)[0][1])

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="female-doctor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    with open("additional_style/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()
    
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("While this approach is generally reliable for predicting outcomes, it doesn't guarantee absolute certainty. It performs well in making predictions but, like any statistical or machine learning model, it operates within a range of probabilities rather than offering definitive outcomes.")
    
    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar(input_data)
        st.plotly_chart(radar_chart)
    
    with col2:
        st.header("The Results Based on Your Report")
        st.subheader("==>")
        add_prediction(input_data)
    
    st.write("Thank you for visiting this web app.")
    st.markdown("<hr>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()

File Structure

data/breast-cancer.csv: The dataset file.
model/model.pkl: The trained logistic regression model.
model/scalar.pkl: The standard scaler used for data transformation.
additional_style/style.css: Custom CSS for the Streamlit app.

Running the Project
To run the project, simply execute the main function. This will clean the data, train the model, and save the trained model and scaler. To use the Streamlit web app, run:

streamlit run <name_of_this_script>.py