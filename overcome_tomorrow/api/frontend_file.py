import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import dateutil.parser
from PIL import Image
from io import StringIO
from utils_api.data_visualization_functions import *
import os

BACKEND_URL = os.environ["BACKEND_URL"]
model_names = requests.get(f'{BACKEND_URL}/models').json()
model_name = model_names[0]
activities = pd.read_json(requests.get(
    f'{BACKEND_URL}/data/activities').json())

st.set_page_config(layout="wide")
# CSS pour changer le fond de l'application / le design des boutons
color_title = "#FE9A2E"
color_text = "#FAFAFA"
size_text = "13"
sport_emojis = {
    "running": "🏃‍♂️",
    "cycling": "🚴",
    "swimming": "🏊‍♂️",
    "walking": "🚶",
    "climbing": "🧗‍♂️",
    "kayaking": "🚣",
    "training": "🏋️‍♂️"
    # Add more mappings as needed
}
css = """
<style>
    .stApp {
        background-image: linear-gradient(to right, #0B4C5F, #1B8A9C); /* Gradient colors */
        color: #FFF; /* General text color */

    }

    .stButton>button {
        color: #001233;  /* Button text color */
        border: none;  /* No border */
        border-radius: 30px;  /* Rounded corners */
        background-color: #FE9A2E;  /* Button background color */
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); /* Box shadow */
        transition: background-color 0.3s, box-shadow 0.3s; /* Animation */
    }

    .stButton>button:hover {
        background-color: #FEC02E; /* Hover color */
        box-shadow: 0 6px 12px 0 rgba(0,0,0,0.3); /* Intensified shadow on hover */
    }

    h1, h2, h3, h4, p {
        font-family: 'Arial', sans-serif; /* Custom font */
    }
}

</style>


"""
st.markdown(css, unsafe_allow_html=True)  # Pour autoriser les commandes html

# API lancée localement pour l'instant, mettre API en ligne ensuite

# DISPLAY MATE
def display_mate():
    image_size = 136

    MATES = {
        'Nicolas B': 'https://res.cloudinary.com/wagon/image/upload/c_fill,g_face,h_200,w_200/v1706536492/hqg5yim3noq0s5mnk17u.jpg',
        'Julien R': 'https://res.cloudinary.com/wagon/image/upload/c_fill,g_face,h_200,w_200/v1706552025/dv5dmvijpyjzkn2zw0lp.jpg',
        'Ruben C': 'https://avatars.githubusercontent.com/u/72013952?v=4',
    }

    MATES_CSS = f"""
    #mates {{
        display: flex;
        justify-content: center; /* Centrer les éléments horizontalement */
        flex-wrap: wrap;
    }}

    .mate-card {{
        display: flex;
        flex-direction: column;
        align-items: center; /* Centrer les éléments verticalement */
    }}

    .mate-card img {{
        width: {image_size}px;
        height: {image_size}px;
        border-radius: 100%;
        padding: 4px;
        margin: 10px;
        box-shadow: 0 0 4px #eee;
    }}

    .mate-card span {{
        text-align: center;
    }}
    """

    MATES_CARD = """\
        <div class="mate-card">
            <img src="{url}">
            <span>{name}</span>
        </div>
    """

    mates = ''.join([MATES_CARD.format(name=f'{name.split()[0]}', url=url) for name, url in MATES.items()])

    MATES_HTML = f"""
    <style>
    {MATES_CSS}
    </style>
    <div id="mates">
        {mates}
    </div>
    """
    st.write(MATES_HTML, unsafe_allow_html=True)

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
    duration = (hours, minutes)
    # st.write(f'test : {response}')
    return calories, sport, distance, avg_power, avg_heart_rate, avg_speed, max_speed, duration, timestamp_dt, start_time_dt


def display_activities_sum_true():
    response = requests.get(
        f'{BACKEND_URL}/activities?activity_datetime=2024-03-04&summarized=true').json()[0]
    return extract_activity_from_json(response)


def predict_next_activity():
    next_activity = json.loads(requests.get(
        f'{BACKEND_URL}/activities/next', params={"models_name": model_name}).json()[0])

    return extract_activity_from_json(next_activity)


def predict_next_activities(selected_models):
    next_activities = requests.get(
        f'{BACKEND_URL}/activities/next', params={"models_name": selected_models}).json()
    res = [json.loads(next_activity) for next_activity in next_activities]
    df = pd.DataFrame.from_dict(res)
    df["total_distance"] = round(df["total_distance"] / 1000, 2)
    df["enhanced_avg_speed"] = round(df["enhanced_avg_speed"] * 3.6, 2)
    df["enhanced_max_speed"] = round(df["enhanced_max_speed"] * 3.6, 2)
    return df


def predict_activity_date(date):
    date_activity = json.loads(requests.get(
        f'{BACKEND_URL}/activities/date?date={date}', params={"model_name": model_name}).json())
    return extract_activity_from_json(date_activity)


def compare_activity_date(date):
    res = requests.get(
        f'{BACKEND_URL}/activities/date/compare', params={"model_name": model_name, "date": date}).json()
    df = pd.read_json(StringIO(res))
    df["total_distance"] = round(df["total_distance"] / 1000, 2)
    df["enhanced_avg_speed"] = round(df["enhanced_avg_speed"] * 3.6, 2)
    df["enhanced_max_speed"] = round(df["enhanced_max_speed"] * 3.6, 2)
    return df


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
    st.markdown(f"<h1 style='text-align: center; color: {color_title};'>Overcome Tomorrow</h1>",
                unsafe_allow_html=True)

    # Pour centrer également le texte sous le titre
    st.markdown(f"<p style='text-align: center;color:{color_text};'>Welcome to Overcome Tomorrow – your ultimate partner in sports performance optimization and activity forecastings. 🌟🏋️‍♂️<br><br>"
                "Harness the power of your health and sports data to unlock personalized activity recommendations and elevate your performance. 💪📈<br><br>"
                "Our platform uses advanced DeepLearning models to forecast your next sports activity, ensuring each suggestion is perfectly tailored to your fitness level and goals. 🧠🏋️‍♀️<br><br>"
                "Begin your journey today and transform your performance for tomorrow. 🚀🌟</p>", unsafe_allow_html=True)


    st.markdown(f"<h4 style='text-align: left; color: {color_title};'>Example of data from one of our athletes :</h1>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: left;color:{color_text};'>Nicolas's marathon run in Chicago</p>",
                unsafe_allow_html=True)

    # On récupère les données qu'on veut grâce à la fonction diplay activities summarized egale True et false
    calories, sport, distance, avg_power, avg_heart_rate, avg_speed, max_speed, duration, timestamp_dt, start_time_dt = display_activities_sum_true()
    data_power, data_stamina, data_speed, data_heart_rate = display_activities_sum_false()
    sport_emoji = sport_emojis.get(sport.lower(), "")
    st.markdown(
        f"<p style='text-align: left;color:{color_text};'>Activity : {sport} {sport_emoji}</p>", unsafe_allow_html=True)

    # Ici les colonnes pour display les infos
    col_info1, col_info2, col_info3 = st.columns([3, 3, 2])
    with col_info1:
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>🏁 Start time : {start_time_dt} </p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>⌛ Finish time : {timestamp_dt} </p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>🕒 Time : {duration[0]}h{duration[1]}</p>", unsafe_allow_html=True)

    with col_info2:
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>📏 Distance : {distance} km</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>⚡ Average Speed : {avg_speed} km/h</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>💨 Max Speed : {max_speed} km/h</p>", unsafe_allow_html=True)

    with col_info3:
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>❤️ Avg Heart Rate : {avg_heart_rate} bpm</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>💪 Avg Power : {avg_power}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>🔥 Calories : {calories} kcal</p>", unsafe_allow_html=True)

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

    st.markdown(f"<h4 style='text-align: left; color: {color_title};'>How it works</h1>",
                unsafe_allow_html=True)
    image = Image.open('overcome_tomorrow/api/static_data/How_it_works.png')
    st.image(image, caption='', use_column_width=False)
    display_mate()
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

        # Display mates
# Page - Begin your journey

def second_page():
    # Accueil de la deuxième page
    st.markdown(f"<h1 style='text-align: Left; color: {color_title};'>Your Journey</h1>",
                unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: left; color: {color_title};'>Predict your next activity :</h1>",
                unsafe_allow_html=True)

    st.markdown(f"<p style='text-align: left;color:{color_text};'>Here you can predict your next activity based on your last 60 activities : </p>",
                unsafe_allow_html=True)

    # Initialisation des variables pour que l'affichage démmarre à 0
    calories, sport, sport_emoji, distance, avg_power, avg_heart_rate, avg_speed, max_speed, duration, timestamp_dt, start_time_dt = (
        0, "Waiting", "", 0, 0, 0, 0, 0, (0, 0), datetime.min, datetime.min)
    global model_name
    model_name = st.selectbox('Select a model to use', model_names)

    if st.button("EVALUATE 📈"):
        # Nouvelles valeurs des variables
        calories, sport, distance, avg_power, avg_heart_rate, avg_speed, max_speed, duration, timestamp_dt, start_time_dt = predict_next_activity()
        sport_emoji = sport_emojis.get(sport.lower(), "")        
        
    st.markdown(
        f"<p style='text-align: left;color:{color_text};'>Activity : {sport} {sport_emoji}</p>", unsafe_allow_html=True)

    col_graph1, col_graph2, col_info = st.columns([3, 3, 2])

    with col_graph1:
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>🏁 Start time : {start_time_dt.time()} </p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>⌛ Finish time : {timestamp_dt.time()} </p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>🕒 Time : {duration[0]}h{duration[1]} </p>", unsafe_allow_html=True)

    with col_graph2:
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>📏 Distance :  {distance} km</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>⚡ Average Speed : {avg_speed} km/h</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>💨 Max Speed : {max_speed} km/h</p>", unsafe_allow_html=True)

    with col_info:
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>❤️ Avg Heart Rate : {round(avg_heart_rate,0)}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>💪 Avg Power: {avg_power}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>🔥 Calories : {round(calories,2)} Kcal</p>", unsafe_allow_html=True)

    st.markdown(f"<h4 style='text-align: left; color: {color_title};'>Predict with a given date :</h1>",
                unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left;color:{color_text};'>Here you can predict an activity with a given date which is
                based on the last 60 activities before this date. You can compare this prediction with your actual performance this day : </p>""",
                unsafe_allow_html=True)
    date = st.date_input(
        "Choose your date 📅")
    calories_date, sport_date, sport_emoji_date, distance_date, avg_power_date, avg_heart_rate_date, avg_speed_date, max_speed_date, duration_date, timestamp_dt_date, start_time_dt_date = (
        0, "Waiting", "", 0, 0, 0, 0, 0, (0, 0), datetime.min, datetime.min)

    if st.button("EVALUATE 📊"):
        # Nouvelles valeurs des variables
        calories_date, sport_date, distance_date, avg_power_date, avg_heart_rate_date, avg_speed_date, max_speed_date, duration_date, timestamp_dt_date, start_time_dt_date = predict_activity_date(
            date)
        sport_emoji_date = sport_emojis.get(sport_date.lower(), "")
    st.markdown(
        f"<p style='text-align: left;color:{color_text};'>Activity : {sport_date}{sport_emoji_date}</p>", unsafe_allow_html=True)

    col_graph1_bis, col_graph2_bis, col_info_bis = st.columns([3, 3, 2])

    with col_graph1_bis:
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>🏁 Start time : {start_time_dt_date.time()} </p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>⌛ Finish time : {timestamp_dt_date.time()} </p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>🕒 Time : {duration_date[0]}h{duration_date[1]} </p>", unsafe_allow_html=True)

    with col_graph2_bis:
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>📏 Distance :  {distance_date} km</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>⚡ Average Speed : {avg_speed_date} km/h</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>💨 Max Speed : {max_speed_date} km/h</p>", unsafe_allow_html=True)

    with col_info_bis:
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>❤️ Avg Heart Rate : {round(avg_heart_rate_date,0)}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>💪 Avg Power: {avg_power_date}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: left;color:{color_text};font-size:{size_text}px;'>🔥 Calories : {round(calories_date,2)} Kcal</p>", unsafe_allow_html=True)

def third_page():
    # Accueil de la page Compare

    st.markdown(f"<h1 style='text-align: Left; color: {color_title};'>Let's compare 🆚</h1>",
                unsafe_allow_html=True)

    st.markdown(f"<h4 style='text-align: left; color: {color_title};'>Compare with a given date :</h1>",
                unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left;color:{color_text};'>Here you can predict an activity with a given date which is
                based on the last 60 activities before this date. You can compare this prediction with your actual performance this day : </p>""",
                unsafe_allow_html=True)

    global model_name
    model_name = st.selectbox('Select a model to use', model_names)

    date = st.date_input("Choose your date 📅")
    df = pd.DataFrame()
    if st.button("COMPARE 📊"):
        df = compare_activity_date(date)

    st.write(df)

# All possible outcom page


def fourth_page():

    st.markdown(f"<h1 style='text-align: Left; color: {color_title};'>Now it's time for you to choose 🤔</h1>",
                unsafe_allow_html=True)

    st.markdown(f"<h2 style='text-align: left; color: {color_title};'>\
                <blockquote> \
                <p>I went forward in time... to view alternate futures. To see all the possible outcomes...</p> \
                </blockquote> \
                <figcaption> \
                <p>Doctor Strange (Avengers: Infinity War, 2018)</p>\
                </figcaption></h2>",
                unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left;color:#CAC0B3;'>You can now select as many model as you please, and for each one of them, predict your performance given you current history activities.</p>""",
                unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left;color:#CAC0B3;'>Then, like Doctor Strange, you will be able to select the futur that suits you best 😀</p>""",
                unsafe_allow_html=True)

    selected_model = st.multiselect(
        "Select models:",
        model_names
    )
    # st.write(selected_model)

    if st.button("PREDICT POSSIBLE ACTIVITES 📊"):
        df = predict_next_activities(",".join(selected_model))
        st.write(df)


# Page - Data analysis


def fifth_page():
    st.markdown(f"<h1 style='text-align: Left; color: {color_title};'>Data Analysis</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: left;color:#CAC0B3;'>Here you can look at some data</p>",
                unsafe_allow_html=True)

    data = activities  # pd.read_csv("raw_data/activities.csv")

    # Define a common text color for titles
    text_color = color_text  # Example: Bright red, adjust as needed

    # Using columns to display two graphs per line and using markdown for colored titles
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<h3 style='color: {text_color};'>Activities per Month</h3>", unsafe_allow_html=True)
        st.pyplot(plot_activities_per_month(data))
    with col2:
        st.markdown(
            f"<h3 style='color: {text_color};'>Sport Distribution</h3>", unsafe_allow_html=True)
        st.pyplot(visualize_sport_distribution(data))

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            f"<h3 style='color: {text_color};'>Monthly Distance Over Years</h3>", unsafe_allow_html=True)
        st.pyplot(plot_monthly_distance_over_years(data))
    with col4:
        st.markdown(
            f"<h3 style='color: {text_color};'>Cadence VS Speed</h3>", unsafe_allow_html=True)
        st.pyplot(plot_scatter_average_cadence_vs_speed(data))

    col5, col6 = st.columns(2)
    with col5:
        st.markdown(
            f"<h3 style='color: {text_color};'>Average Speed Over Time</h3>", unsafe_allow_html=True)
        st.pyplot(plot_average_speed_over_time(data))
    with col6:
        st.markdown(
            f"<h3 style='color: {text_color};'>Monthly Calorie and Distance</h3>", unsafe_allow_html=True)
        st.pyplot(plot_monthly_calorie_and_distance(data))

    # Input widgets for graphs requiring specific parameters, with titles also in color
    st.markdown(
        f"<h3 style='color: {text_color};'>Monthly Distance for a Specific Year and Month</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Enter the year", value=2023,
                               step=1, min_value=2019, max_value=2024)
    with col2:
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        month_ = st.selectbox("Select the month", options=months, index=0)
        month = months.index(month_) + 1
    st.pyplot(plot_monthly_distance(year, month, data))

    
    st.markdown(
        f"<h3 style='color: {text_color};'>Analysis of Last Activities for a Specific Sport</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:

        num_days = st.number_input(
            "Enter the number of days", min_value=1, max_value=100, value=20)
    with col2:
        # Use st.selectbox for sport type selection with titles in color
        sport_type_options = ["running", "cycling",
                              "walking", "training", "swimming"]
        sport_type = st.selectbox(
            "Select the sport type", options=sport_type_options)

    st.pyplot(analyze_last_activities(data, num_days, sport_type))


# Navigation
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

# Sidebar
with st.sidebar:
    if st.button("Home Page 🏠"):
        st.session_state['current_page'] = 'home'
    if st.button('Begin your journey 🏔️'):
        st.session_state['current_page'] = 'second'
    if st.button('Compare predict VS real 🔎'):
        st.session_state['current_page'] = 'third'
    if st.button('Decide your futur 🎯'):
        st.session_state['current_page'] = 'fourth'
    if st.button('Data analysis 📊'):
        st.session_state['current_page'] = 'fifth'

if st.session_state['current_page'] == 'home':
    home_page()
elif st.session_state['current_page'] == 'second':
    second_page()
elif st.session_state['current_page'] == 'third':
    third_page()
elif st.session_state['current_page'] == 'fourth':
    fourth_page()
elif st.session_state['current_page'] == 'fifth':
    fifth_page()
