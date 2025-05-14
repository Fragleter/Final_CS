##This file contains the function to display care recommendations for the recommended crop (i.e. the section which appears below the crop recommendation).
##Taking data (information about plant care) from the JSON file ("plant_care_database.json"), which was gathered from both a gardening app (Greg app) and AI (ChatGPT), we wanted to provide the user with additional tips for the crop they were recommended to plant.
##This serves as an additional section, not as output from the machine learning model prediction.

import json ##Used to load the JSON file containing the plant care data. The JSON file consists of general care tips and recommendations, found through internet research and typed in manually.
import os ##Used to access functions, navigating through the file system.

import streamlit as st #Used for creating and visualising the web app

##The structure (or "template") was taken from ChatGPT and was filled in manually, recommendations from Github Copilot were also used.
##Loading the JSON file containing the plant care data
def load_care_data():
    try:
        if os.path.exists("plant_care_database.json"):
            with open("plant_care_database.json", "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            st.error("The file plant_care_database.json cannot be found.")
            return {}

    except Exception as e:
        st.error(f"Error during care data loading: {e}")
        return {}

##Displaying the additional section for care recommendations and tips
def display_care_recommendations(crop, temperature, humidity):
    care_data = load_care_data()        ##Function load_care_data() loads the plant care data/info from the JSON file and stores it in care_data (variable)
    crop_lower = crop.lower()       ##Used to convert crop names to lowercase for consistency (match what's in the JSON file)

    care_info = care_data.get(crop_lower, {})       ##Gets the care information for the given crop, if nothing found, returns an empty dictionary (display of an error message, see hereunder)
    if not care_info:
        st.error(f"No care recommendation found for **{crop}**")
        return

##WATERING
    watering_info = care_info.get("watering", {})

    frequency = watering_info.get("frequency", "No frequency info available.")
    amount = watering_info.get("amount", "No amount info available.")
    tips = watering_info.get("tips", "No tips available.")


    st.subheader(f"**{crop}**")

    ##Display for the user
    st.markdown("### Watering💧")
    st.write(f"**Frequency:** {frequency}")
    st.write(f"**Amount:** {amount}")
    st.write(f"**Tips:** {tips}")
    st.write("") #(Spacing)

##SUNLIGHT
    sunlight_info = care_info.get("sunlight", {})

    sun_hours= sunlight_info.get("frequency", "No sunlight hours info available.")
    sun_type = sunlight_info.get("amount", "No sunlight type info available.")
    sun_tips = sunlight_info.get("tips", "No sunlight tips available.")

    ##Display for the user
    st.markdown("### Sunlight☀️")
    st.write(f"**Hours per day:** {sun_hours}")
    st.write(f"**Amount:** {sun_type}")
    st.write(f"**Tips:** {sun_tips}")
    st.write("") ##(Spacing)

##PLANTING
    planting_info = care_info.get("planting_time", {})

    when = planting_info.get("when", "No frequency info available.")

    ##Display for the user
    st.markdown("### Planting🌱")
    st.write(f"**Recommended planting period:** {when}")




