##This file contains the main code for our app.
##Idea: As sustainability and health are becoming of interest to more and more people, we wanted to create an app that would help users get into gardening, growing their own fruits and vegetables, in a simple and effective way.
##How it works: Based on the input provided by the user, the app provides the best-suited crop recommendation. Additional general information and tips are provided for the recommended crop.
##With the given location, we find the current weather data (temperature and humidity) using the Open Meteo API and recommend crops accordingly.

import datetime as dt ##Used to get the current date and time
import pickle ##Used to save and load the machine learning model
import numpy as np ##Used for numerical operations (mathematical part)
import requests ##Used for HTTP requests to the Open Meteo API for the necessary weather data
import streamlit as st ##Used for creating and visualizing the web app

from kl import display_care_recommendations ##Importing the functions to display care recommendations (after output, i.e. recommended crop)

##Resources and help:
## - CS Coaching tutors: help with finding datasets, checking whether they are usable for our project, and so on.
## - Friends and relatives: help with code structure, debugging and fixing mistakes or problems.
## - AI:
##       - ChatGPT and Claude: help with code structure, debugging and corrections or improvements, finding functions or code snippets to use for what we wanted to implement, research on plant care information and tips, and so on.
##       - Github Copilot: automatic suggestions, improving code, fixing problems and errors, and so on.
## - Greg.app: Research and information to use for the plant care recommendations and tips section (see kl.py and plant_care_database.json).
## - Kaggle: finding datasets and examples of similar projects.
## - Open Meteo API: getting the weather data (current temperature and humidity) based on the user's location.


##Front/welcome page display
st.set_page_config(page_title="Local weather and crop recommendation", page_icon="🌱", layout="centered")

st.title("Smart Garden Planner 🥬")
st.subheader("Welcome to the Smart Garden Planner! This app helps you plan your garden efficiently.")
st.write("Plan your dream garden by finding the best crop for you, based your location as well as current weather conditions and plant data!")
st.markdown("**Please choose your location in the sidebar.**")


##User input: form to allow user to provide their location, city and country (input)
with st.sidebar:
    st.header("Choose your location")
    with st.form(key="city_form", clear_on_submit=False):
        city_input = st.text_input("City", placeholder="ex. Paris")
        country_input = st.text_input("Country (optional)", placeholder="ex. France")
        submitted = st.form_submit_button("Search")


##"search_results" contains the list of results from the geocoding API, meaning the list of countries and cities available
##"selection" stores the index of the selected location from the list of results
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "selection" not in st.session_state:
    st.session_state.selection = None


##Geocoding API call
if submitted:
    if not city_input:
        st.sidebar.error("Please enter a city name.")       ##Display of error message (in "") if the user has not entered a city name
    else:
        query = f"{city_input} {country_input}".strip()
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        try:
            geo_resp = requests.get(geo_url, params={"name": query, "count": 5, "language": "en"}, timeout=8)       ##Sending a GET request to the geocoding API with the user's input (top 5 results, in english, waiting up to 8 seconds)
            geo_resp.raise_for_status()         ##Checking if the request was successful, raising an error if the HTTP request has failed.
            results = geo_resp.json().get("results", [])
            if not results:
                st.sidebar.error("No results found. Please try again.")
            else:   ##If we have a result:
                st.session_state.search_results = results       ##Saving the results
                st.session_state.selection = 0 if len(results) == 1 else None       ##If only one result, select it, and otherwise, no selection yet.
        except Exception as e:
            st.sidebar.error(f"Geocoding error: {e}")       ##For errors occurring during the geocoding API call.

##User selection of the location
selected_location = None
if st.session_state.search_results:         ##Checking for the list of results from the geocoding API
    results = st.session_state.search_results
    if len(results) == 1:       ##If exactly one result: the result is automatically selected.
        selected_location = results[0]
    else:       ##If multiple results: selectbox to allow the user to choose one of them
        labels = [f"{r['name']}, {r.get('admin1','')} ({r.get('country','')})" for r in results]        ##For simplification, labels are created with the name of the city, the state (if available) and the country, with aim to help the user distinguish between the different results (e.g. Geneva, Switzerland vs. Geneva, USA).
        idx = st.sidebar.selectbox("Multiple results found: please choose!", options=list(range(len(labels))), format_func=lambda i: labels[i], index=st.session_state.selection if st.session_state.selection is not None else 0)
        st.session_state.selection = idx        ##Storing user selection
        selected_location = results[idx]        ##Settting the location selected by the user as selected_location.

##Getting weather data and crop recommendation
if selected_location:
    lat = selected_location["latitude"]
    lon = selected_location["longitude"]
    display_name = f"{selected_location['name']}, {selected_location.get('country', '')}".strip(', ')

    st.sidebar.success(f"📌 {display_name}\n({lat:.4f}, {lon:.4f})")         ##Display of successfully selected location (latitude and longitude) in the sidebar, "4f" meaning that we're collecting latitude and longitude with 4 decimal places.

    ##Call to the weather API to get the current weather data
    BASE_URL = "https://api.open-meteo.com/v1/forecast"         ##URL for weather API
    params = {      ##Dictionary of parameters to send with the API request: location (lat and lon), current temperature and humidity and time zone (lets the API choose the correct timezone).
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m",
        "timezone": "auto",
    }
    try: ##Handling potential errors
        weather_resp = requests.get(BASE_URL, params=params, timeout=10)        ##Sending request to API with chosen parameters (see above), waiting up to 10 seconds.
        weather_resp.raise_for_status()         ##Raising error if HTTP request fails.
        data = weather_resp.json()      ##COnverting response to JSON (easier to use).
    except Exception as err:
        st.error(f"Weather forecast error: {err}")
        st.stop()

    current = data.get("current", {})       ##Getting current weather data from the response or empty if not found.
    ##Getting current temperature and relative humidity from weather data
    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    ##Same for current time, formatted as day, month, year, hour, minute and second ("%d%b%Y%H:%M:%S").
    time_retrieved = dt.datetime.now().strftime("%d%b%Y%H:%M:%S")

    ##Displaying current weather (temperature and humidity)
    col1, col2 = st.columns(2)      ##Creating to columns to display temperature and humidity in streamlit.
    with col1:
        st.metric("Temperature", f"{temperature}°C")
    with col2:
        st.metric("Humidity", f"{humidity}%")

    ##Loading the pretrained machine learning model for crop recommendation
    try:
        with open("data/data2/crop_rf_model.pkl", "rb") as f:       ##Reading the model from the crop file ("rb" = "read binary", reading in binary mode).
            crop_model = pickle.load(f)         ##Loading the machine learning model (using pickle)
    except Exception as e:      #If problem loading the model
        st.error(f"Impossible to load the model (crop_rf_model.pkl): {e}")
        st.stop()

    features = np.array([[temperature, humidity]])      ##Shape (1,2), sample with 2 features (temperature and humidity)
    try:
        crop_pred = crop_model.predict(features)[0]         ##Using the loaded ML model to predict the crop based on input (temperature and humidity), [0] to get the first element of the array (the predicted crop).
    except Exception as e:
        st.error(f"Error occurred during prediction: {e}")
        st.stop()

    ##Displaying the crop recommendation
    st.subheader("Recommended crop")
    st.success(f"💡{crop_pred}")


##Graph: scatter plot with the predicted crop
    def plot_scatter_with_prediction(user_temp, user_humidity, crop_pred,
                                     dataset_path="data/data2/Crop_Recommendation.csv"):    ##Scatter plot using user inputs and crop dataset
        import pandas as pd
        import plotly.express as px

        df = pd.read_csv(dataset_path) ##Loads data from CSV file into dataframe (contains “Temperature”, ‘Humidity’, “Crop” columns)

        ##Selection of x and y data
        x = df["Temperature"]
        y = df["Humidity"]
        crop_types = df["Crop"]

        ##Creation of the scatter plot (using plotly express, creating interactive scatter plot)
        fig = px.scatter(
            df,     ##data used
            x=x,    ##Variables on x and y axis
            y=y,
            color="Crop",   ##each point is colored according to Crop
            opacity=0.3,    ##points are semi-transparent to better see overlapping ones
            labels={"Temperature": "Temperature (°C)", "Humidity": "Humidity (%)"}, ##axis names
            title="Relationship between temperature and humidity by crop" ##title of the scatterplot graph
        )

        ##Adds the user's prediction point to the graph
        fig.add_scatter(
            x=[user_temp],      ##Coordinates of the point
            y=[user_humidity],
            mode="markers",     ##Adds just a point (not a line)
            marker=dict(color="red", size=12, symbol="star"),   ##Point customization (red so it stands out more, bigger size than the other points, star shaped to distinguish it)
            name=f"Prédiction: {crop_pred}"     ##Sets the label that will appear in the legend for the predicted crop point
        )

        return fig  ##Calls function with user values, and stores graph in fig


    fig = plot_scatter_with_prediction(temperature, humidity, crop_pred)    ##Calls the function to generate and return a scatter plot showing the relationship between temperature and humidity for various crops, highlighting the user's prediction on the graph
    st.plotly_chart(fig)    ##Display the interactive graphic in Streamlit app

    st.caption(f"Update: {time_retrieved}")     ##Displays a legend below with the date/time when the prediction was made


##Display of plant care recommendations and tips (only if crop_pred, i.e. the recommended crop, is defined)
if 'crop_pred' in locals():
    st.subheader("Plant care recommendations and tips👨🏽‍🌾")
    display_care_recommendations(crop=crop_pred, temperature=temperature, humidity=humidity) ##See kl.py


