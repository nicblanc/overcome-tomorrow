import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import dateutil.parser
from PIL import Image
import os

BACKEND_URL = os.environ["BACKEND_URL"]

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
st.markdown(css, unsafe_allow_html=True)  # Pour autoriser les commandes html

# API lancée localement pour l'instant, mettre API en ligne ensuite


def extract_activity_from_json(json_response):
    calories = json_response.get('total_calories')
    sport = json_response.get('sport').upper()
    distance = round(json_response.get('total_distance')/1000, 2)
    avg_power = json_response.get('avg_power')
    avg_heart_rate = json_response.get('avg_heart_rate')
    avg_speed = round(json_response.get('enhanced_avg_speed')*3.6, 2)
    max_speed = round(json_response.get('enhanced_max_speed')*3.6, 2)
    timestamp = json_response.get('timestamp')
    start_time = json_response.get('start_time')
    timestamp_dt = dateutil.parser.parse(timestamp)
    start_time_dt = dateutil.parser.parse(start_time)

    # Calcul de la différence en heures et minutes
    diff = timestamp_dt - start_time_dt

    # Extraction des heures et minutes
    hours = diff.seconds // 3600
    minutes = (diff.seconds % 3600) // 60
    time = (hours, minutes)
    # st.write(f'test : {response}')
    return calories, sport, distance, avg_power, avg_heart_rate, avg_speed, max_speed, time


def display_activities_sum_true():
    response = requests.get(
        f'{BACKEND_URL}/activities?activity_datetime=2024-03-04&summarized=true').json()[0]
    return extract_activity_from_json(response)


def predict_next_activity():
    next_activity = json.loads(requests.get(
        f'{BACKEND_URL}/activities/next').json()[0])
    print(next_activity)
    return extract_activity_from_json(next_activity)


def predict_activity_date(date):
    date_activity = json.loads(requests.get(
        f'{BACKEND_URL}/activities/date?date={date}').json())
    return extract_activity_from_json(date_activity)


def display_activities_sum_false():
    response_false = json.loads(requests.get(
        f'{BACKEND_URL}/activities?activity_datetime=2024-03-04&summarized=false').json()[0])
    data_power = response_false['power']
    data_stamina = response_false['137']
    data_speed = response_false['enhanced_speed']
    data_heart_rate = response_false['heart_rate']
    return data_power, data_stamina, data_speed, data_heart_rate

# HomePage - Overcome Tomorrow


def home_page():
    # Centrer le titre et ajuster le style avec HTML et CSS
    st.markdown("<h1 style='text-align: center; color: #FF595A;'>Overcome Tomorrow</h1>",
                unsafe_allow_html=True)

    # Pour centrer également le texte sous le titre
    st.markdown("<p style='text-align: center;color:#CAC0B3;'>Welcome to Overcome Tomorrow – your ultimate partner in sports performance optimization and activity forecastings. <br><br>"
                "Harness the power of your health and sports data to unlock personalized activity recommendations and elevate your performance.<br><br>"
                "Our platform uses advanced DeepLearning models to forecast your next sports activity, ensuring each suggestion is perfectly tailored to your fitness level and goals.<br><br>"
                "Begin your journey today and transform your performance for tomorrow.</p>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: left; color: #FF595A;'>Example of data from one of our athletes :</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: left;color:#CAC0B3;'>Nicolas's marathon run in Chicago</p>",
                unsafe_allow_html=True)

    # On récupère les données qu'on veut grâce à la fonction diplay activities summarized egale True et false
    calories, sport, distance, avg_power, avg_heart_rate, avg_speed, max_speed, time = display_activities_sum_true()
    data_power, data_stamina, data_speed, data_heart_rate = display_activities_sum_false()
    st.markdown(
        f"<p style='text-align: left;color:#CAC0B3;'>Activity : {sport}</p>", unsafe_allow_html=True)

    # Ici les colonnes pour display les infos
    col_info1, col_info2, col_info3 = st.columns([3, 3, 2])
    with col_info1:
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Distance : {distance} km</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Time : {time[0]}h{time[1]}</p>", unsafe_allow_html=True)

    with col_info2:
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Average Speed : {avg_speed} km/h</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Max Speed : {max_speed} km/h</p>", unsafe_allow_html=True)

    with col_info3:
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Avg Power : {avg_power} Watt</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Avg Heart Rate : {avg_heart_rate} bpm</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;font-size:16px;'>Calories : {calories} kcal</p>", unsafe_allow_html=True)

    # Ici les colonnes pour display les graphiques
    col_graph1, col_graph2, col_graph3 = st.columns([3, 0.5, 3])

    with col_graph1:
        # Calcul du temps timestamp
        timestamps = list(data_power.keys())
        first_timestamp = int(timestamps[0]) / 1000  # Convertir en secondes
        time_seconds = [(int(ts) / 1000) -
                        first_timestamp for ts in timestamps]
        option = st.selectbox('', ['Power', 'Stamina', 'Speed', 'Heart Rate'])
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
        plt.plot(time_seconds, display, linestyle='-',
                 color='tab:orange', linewidth=2, label=f'{option}')
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


    st.markdown("<h4 style='text-align: left; color: #FF595A;'>How it works</h1>",
                unsafe_allow_html=True)
    image = Image.open('overcome_tomorrow/api/static_data/How_it_works.png')
    st.image(image, caption='How it works', use_column_width=False)

    with col_graph3:
        option_2 = st.selectbox(
            '', ['Stamina', 'Power', 'Speed', 'Heart Rate'])
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
        plt.plot(time_seconds, display_2, linestyle='-',
                 color='tab:orange', linewidth=2, label=f'{option_2}')
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
    st.markdown("<h1 style='text-align: Left; color: #FF595A;'>Your Journey</h1>",
                unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: left; color: #FF595A;'>Predict your next activity :</h1>",
                unsafe_allow_html=True)

    st.markdown("<p style='text-align: left;color:#CAC0B3;'>Here you can predict your next activity based on your last 60 activities : </p>",
                unsafe_allow_html=True)

    # Initialisation des variables pour que l'affichage démmarre à 0
    calories, sport, distance, avg_power, avg_heart_rate, avg_speed, max_speed, time = (
        0, "En attente", 0, 0, 0, 0, 0, (0, 0))

    if st.button("EVALUATE 📈"):
        # Nouvelles valeurs des variables
        calories, sport, distance, avg_power, avg_heart_rate, avg_speed, max_speed, time = predict_next_activity()

    st.markdown(
        f"<p style='text-align: left;color:#CAC0B3;'>Activity : {sport}</p>", unsafe_allow_html=True)

    col_graph1, col_graph2, col_info = st.columns([3, 3, 2])

    with col_graph1:
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Distance :  {distance} km</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Time : {time[0]}h{time[1]} </p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Avg Heart Rate : {round(avg_heart_rate,0)}</p>", unsafe_allow_html=True)

    with col_graph2:
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Average Speed : {avg_speed} km/h</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Max Speed : {max_speed} km/h</p>", unsafe_allow_html=True)

    with col_info:
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Avg Power: {avg_power} Watt</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Calories : {round(calories,2)} Kcal</p>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: left; color: #FF595A;'>Predict with a given date :</h1>",
                unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left;color:#CAC0B3;'>Here you can predict an activity with a given date which is
                based on the last 60 activities before this date. You can compare this prediction with your actual performance this day : </p>""",
                unsafe_allow_html=True)
    date = st.date_input(
        "Choose your date 📅")
    calories_date, sport_date, distance_date, avg_power_date, avg_heart_rate_date, avg_speed_date, max_speed_date, time_date = (
        0, "En attente", 0, 0, 0, 0, 0, (0, 0))

    if st.button("EVALUATE 📊"):
        # Nouvelles valeurs des variables
        calories_date, sport_date, distance_date, avg_power_date, avg_heart_rate_date, avg_speed_date, max_speed_date, time_date = predict_activity_date(
            date)

    col_graph1_bis, col_graph2_bis, col_info_bis = st.columns([3, 3, 2])

    with col_graph1_bis:
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Distance :  {distance_date} km</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Time : {time_date[0]}h{time_date[1]} </p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Avg Heart Rate : {round(avg_heart_rate,0)}</p>", unsafe_allow_html=True)

    with col_graph2_bis:
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Average Speed : {avg_speed_date} km/h</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Max Speed : {max_speed_date} km/h</p>", unsafe_allow_html=True)

    with col_info_bis:
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Avg Power: {avg_power_date} Watt</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:#CAC0B3;'>Calories : {round(calories_date,2)} Kcal</p>", unsafe_allow_html=True)


# Page - Data analysis


def third_page():
    st.markdown("<h1 style='text-align: Left; color: #FF595A;'>Data Analysis</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: left;color:#CAC0B3;'>Here you can look at some datas</p>",
                unsafe_allow_html=True)
    st.write("(Add some charts about data analysis)")


# Navigation
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

# Sidebar
with st.sidebar:
    if st.button("Home Page 🏠"):
        st.session_state['current_page'] = 'home'
    if st.button('Begin your journey 🏔️'):
        st.session_state['current_page'] = 'second'
    if st.button('Data analysis 📊'):
        st.session_state['current_page'] = 'third'

if st.session_state['current_page'] == 'home':
    home_page()
elif st.session_state['current_page'] == 'second':
    second_page()
elif st.session_state['current_page'] == 'third':
    third_page()
