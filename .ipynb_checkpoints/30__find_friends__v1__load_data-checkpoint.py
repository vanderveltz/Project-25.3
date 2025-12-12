import streamlit as st
import pandas as pd  # type: ignore

DATA = 'welcome_survey_simple_v1.csv'

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')

    return all_df

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])


st.write("Wybrane dane:")
st.dataframe(person_df, hide_index=True)

all_df = get_all_participants()
st.write("Przykładowe osoby z bazy:")
st.dataframe(all_df.sample(10), hide_index=True)
