import streamlit as st
import numpy as np
import pandas as pd 
import pickle as pickl
import plotly.graph_objects as go

def clean_data():
      data=pd.read_csv("dataset/breast-cancer.csv")
      data =data.drop('id',axis=1)
      data['diagnosis']=data['diagnosis'].map({'B':0,'M':1})
    
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
    st.sidebar.header("cell measurements")
    data=clean_data()
    input_dict={}
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
    for label,key in slider_labels:
        input_dict[key]=st.sidebar.slider(
           label,
          min_value=float(0),
          max_value=float(data[key].max()),
          value=float(data[key].mean())
         )
    return input_dict    
def add_prediction(input_data) :
    model=pickl.load(open("prediction_model/model.pkl","rb"))
    scalar=pickl.load(open("prediction_model/scalar.pkl","rb"))
    input_array =np.array(list(input_data.values())).reshape(1,-1)#this is for getting a 2d array
    scaled_array=scalar.transform( input_array)
    prediction=model.predict(scaled_array)

    if(prediction[0]==0):
        st.write("benign")
    else:
        st.write("mallacious")
    st.write("Probability of being benign: ", model.predict_proba(scaled_array)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(scaled_array)[0][1])

def main():
    st.set_page_config(
        page_title="breast cancer predictor",
        page_icon="female-doctor",
        layout="wide",initial_sidebar_state="expanded"

    )
    with open("additonal_style/style.css") as f:
       st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    input_data=add_sidebar()
    
    with st.container():
        st.title("breast cancer predictor")
        st.write("While this approach is generally reliable for predicting outcomes, it doesn't guarantee absolute certainty. It performs well in making predictions but, like any statistical or machine learning model, it operates within a range of probabilities rather than offering definitive outcomes.")
    col1,col2=st.columns([4,1])

    with col1:
        radar_chart = get_radar(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        st.header("The Results based on your report")
        st.subheader("==>")
        add_prediction(input_data)  
    st.write("lots of thanks for visiting this web app")
    st.markdown("<hr>", unsafe_allow_html=True)
              

if __name__=='__main__':
    main()