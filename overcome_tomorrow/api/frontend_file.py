import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json


st.set_page_config(layout="wide")
# CSS pour changer le fond de l'application / le design des boutons
css = """
<style>
    .stApp {
        background-color: #001233;
        background-size: cover;
    }
    .stButton>button {
        color: #CAC0B3;  /* Couleur du texte du bouton */
        border-color: #CAC0B3;  /* Couleur de la bordure du bouton */
        border-width: 2px;  /* Épaisseur de la bordure du bouton */
        border-radius: 20px;  /* Arrondissement des coins du bouton */
        background-color: #001233;  /* Couleur de fond du bouton */
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True) #Pour autoriser les commandes html

# API lancée localement pour l'instant, mettre API en ligne ensuite
def display_activities_sum_true():
    response = requests.get(f'http://0.0.0.0:8000/activities?activity_datetime=2024-03-04&summarized=true').json()[0]
    calories = response['total_calories'][0]
    sport = response['sport'][0].upper()
    distance = round(response['total_distance'][0]/1000,2)
    avg_power = response['avg_power'][0]
    avg_heart_rate = response['avg_heart_rate'][0]
    avg_speed = round(response['enhanced_avg_speed'][0]*3.6,2)
    max_speed = round(response['enhanced_max_speed'][0]*3.6,2)
    timestamp = response['timestamp'][0]
    start_time = response['start_time'][0]
    timestamp_dt = datetime.strptime(timestamp, "%d/%m/%Y %H:%M")
    start_time_dt = datetime.strptime(start_time, "%d/%m/%Y %H:%M")
    # Calcul de la différence en heures et minutes
    diff = timestamp_dt - start_time_dt

    # Extraction des heures et minutes
    hours = diff.seconds // 3600
    minutes = (diff.seconds % 3600) // 60
    time = (hours,minutes)
    #st.write(f'test : {response}')
    return calories, sport, distance, avg_power, avg_heart_rate, avg_speed, max_speed, time

def display_activities_sum_false():
    response_false = json.loads(requests.get(f'http://0.0.0.0:8000/activities?activity_datetime=2024-03-04&summarized=false').json()[0])
    data_power = response_false['power']
    data_stamina = response_false['137']
    data_speed = response_false['enhanced_speed']
    data_heart_rate = response_false['heart_rate']
    return data_power, data_stamina, data_speed, data_heart_rate

# HomePage - Overcome Tomorrow
def home_page():
    # Centrer le titre et ajuster le style avec HTML et CSS
    st.markdown("<h1 style='text-align: center; color: #FF595A;'>Overcome Tomorrow</h1>", unsafe_allow_html=True)

    # Pour centrer également le texte sous le titre
    st.markdown("<p style='text-align: center;color:#CAC0B3;'>Welcome to Overcome Tomorrow – your ultimate partner in sports performance optimization and activity forecastings. <br><br>"
                "Harness the power of your health and sports data to unlock personalized activity recommendations and elevate your performance.<br><br>"
                "Our platform uses advanced DeepLearning models to forecast your next sports activity, ensuring each suggestion is perfectly tailored to your fitness level and goals.<br><br>"
                "Begin your journey today and transform your performance for tomorrow.</p>", unsafe_allow_html=True)

    # Utiliser st.columns pour centrer le bouton
    col1, col2, col3 = st.columns([3,2,2])
    with col2:
        if st.button('BEGIN YOUR JOURNEY 🏔️', key="begin_journey"):
            st.session_state['current_page'] = 'second'

    st.markdown("<h4 style='text-align: left; color: #FF595A;'>Example of insights from one of our athletes :</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left;color:#CAC0B3;'>Nicolas's afternoon run prediction</p>", unsafe_allow_html=True)

    # On récupère les données qu'on veut grâce à la fonction diplay activities summarized egale True et false
    calories, sport, distance, avg_power, avg_heart_rate, avg_speed, max_speed, time = display_activities_sum_true()
    data_power, data_stamina, data_speed, data_heart_rate = display_activities_sum_false()
    st.markdown(f"<p style='text-align: left;color:#CAC0B3;'>Activity : {sport}</p>", unsafe_allow_html=True)

    # Ici les colonnes pour display les infos
    col_info1, col_info2, col_info3 = st.columns([3,3,2])
    with col_info1:
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Distance : {distance} km</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Time : {time[0]}h{time[1]}</p>", unsafe_allow_html=True)

    with col_info2:
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Average Speed : {avg_speed} km/h</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Max Speed : {max_speed} km/h</p>", unsafe_allow_html=True)


    with col_info3:
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Avg Power : {avg_power} Watt</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Avg Heart Rate : {avg_heart_rate} bpm</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Calories : {calories} kcal</p>", unsafe_allow_html=True)

    # Ici les colonnes pour display les graphiques
    col_graph1, col_graph2, col_graph3 = st.columns([3,0.5,3])

    with col_graph1:
        #Calcul du temps timestamp
        timestamps = list(data_power.keys())
        first_timestamp = int(timestamps[0]) / 1000  # Convertir en secondes
        time_seconds = [(int(ts) / 1000) - first_timestamp for ts in timestamps]
        option = st.selectbox('', ['Power','Stamina','Speed','Heart Rate'])
        if option == 'Power':
            display = list(data_power.values())
        elif option == 'Stamina':
            display = list(data_stamina.values())
        elif option == 'Speed':
            display = list(data_speed.values())
        elif option == 'Heart Rate':
            display = list(data_heart_rate.values())

        # Création du graphique
        plt.figure(figsize=(14, 9))
        plt.plot(time_seconds, display, linestyle='-', color='tab:orange', linewidth=2, label=f'{option}')
        plt.title(f'{option} vs. Time', fontsize=16, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel(f'{option}', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        # Afficher le graphique dans Streamlit avec st.pyplot()
        st.pyplot(plt)

    with col_graph3:
        option_2 = st.selectbox('', ['Stamina','Power','Speed','Heart Rate'])
        if option_2 == 'Power':
            display_2 = list(data_power.values())
        elif option_2 == 'Stamina':
            display_2 = list(data_stamina.values())
        elif option_2 == 'Speed':
            display_2 = list(data_speed.values())
        elif option_2 == 'Heart Rate':
            display_2 = list(data_heart_rate.values())

        # Création du graphique
        plt.figure(figsize=(14, 9))
        plt.plot(time_seconds, display_2, linestyle='-', color='tab:orange', linewidth=2, label=f'{option_2}')
        plt.title(f'{option_2} vs. Time', fontsize=16, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel(f'{option_2}', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        # Afficher le graphique dans Streamlit avec st.pyplot()
        st.pyplot(plt)

# Page - Begin your journey
def second_page():
    # Accueil de la deuxième page
    st.markdown("<h1 style='text-align: Left; color: #FF595A;'>Your Journey</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left;color:#CAC0B3;'>Here you can predict your next activity</p>", unsafe_allow_html=True)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.write("(INPUT RESTE  DEFINIR)")
    #uploaded_file = st.file_uploader("Drag and drop your CSV file to generate your results", type="csv")

    #if uploaded_file is not None:
        #data = pd.read_csv(uploaded_file)
        #st.write(data)

    if st.button("EVALUATE 📈"):
        pass # RELIER ICI A L'API PREDICT QUAND ON AURA LE MODEL

    st.markdown(f"<p style='text-align: left;color:#CAC0B3;'>Activity : </p>", unsafe_allow_html=True)

    col_graph1, col_graph2, col_info = st.columns([3,3,2])
    with col_graph1:
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;'>Distance : </p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;'>Time : </p>", unsafe_allow_html=True)


    with col_graph2:
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;'>Average Speed : </p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;'>Max Speed : </p>", unsafe_allow_html=True)

    with col_info:
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;'>Max Power: </p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: left;color:#CAC0B3;'>Calories : </p>", unsafe_allow_html=True)

# Page - Data analysis
def third_page():
    st.markdown("<h1 style='text-align: Left; color: #FF595A;'>Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left;color:#CAC0B3;'>Here you can look at some datas</p>", unsafe_allow_html=True)
    st.write("(Add some charts about data analysis)")

# Navigation
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

if st.session_state['current_page'] == 'home':
    home_page()
elif st.session_state['current_page'] == 'second':
    second_page()
elif st.session_state['current_page'] == 'third':
    third_page()

# Sidebar
with st.sidebar:
    if st.button("Home Page"):
        st.session_state['current_page'] = 'home'
    if st.button('Your journey'):
        st.session_state['current_page'] = 'second'
    if st.button('Data analysis'):
        st.session_state['current_page'] = 'third'
