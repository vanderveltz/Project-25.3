import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px
import plotly.graph_objects as go

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

st.set_page_config(page_title="Twoje Plemię ❤️", layout="wide")

# ------------------------------
# LOADING
# ------------------------------
@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants(_model):
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(_model, data=all_df)
    return df_with_clusters

# ------------------------------
# SIDEBAR UI
# ------------------------------
with st.sidebar:
    st.header("✨ Powiedz nam coś o sobie!")
    st.caption("Pomożemy Ci znaleźć osoby podobne do Ciebie 🙌")

    age = st.selectbox("🎂 Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'])
    edu_level = st.selectbox("🎓 Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("🐾 Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("🏞️ Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("🚻 Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }])

# ------------------------------
# LOAD MODEL AND DATA
# ------------------------------
model = get_model()
all_df = get_all_participants(model)
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

# ------------------------------
# PREDICTION
# ------------------------------
predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
cdata = cluster_names_and_descriptions[predicted_cluster_id]

st.success(f"🎉 **Gotowe! Pasujesz do grupy _{cdata['name']}_** 🎉")
st.markdown(f"### 🧬 Opis Twojej grupy\n{cdata['description']}")

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

# ------------------------------
# METRICS
# ------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("👥 Liczebność grupy", len(same_cluster_df))
with col2:
    st.metric("📊 Liczba klastrów", len(cluster_names_and_descriptions))
with col3:
    st.metric("🎯 Twój klaster", predicted_cluster_id)

st.balloons()
st.divider()

# ------------------------------
# USER PROFILE SUMMARY CARD
# ------------------------------
st.subheader("🧑‍💼 Twój profil")
profile_cols = st.columns(5)
profile_cols[0].info(f"**Wiek:** {age}")
profile_cols[1].info(f"**Wykształcenie:** {edu_level}")
profile_cols[2].info(f"**Zwierzęta:** {fav_animals}")
profile_cols[3].info(f"**Miejsce:** {fav_place}")
profile_cols[4].info(f"**Płeć:** {gender}")

st.divider()

# ------------------------------
# CHARTS: DISTRIBUTIONS
# ------------------------------
st.header("📊 Rozkłady w grupie")

colA, colB = st.columns(2)

with colA:
    fig_age = px.histogram(same_cluster_df.sort_values("age"), x="age", color="age")
    fig_age.update_layout(title="🎂 Wiek w grupie")
    st.plotly_chart(fig_age, use_container_width=True)

with colB:
    fig_edu = px.pie(same_cluster_df, names="edu_level", title="🎓 Wykształcenie w grupie")
    st.plotly_chart(fig_edu, use_container_width=True)

colC, colD = st.columns(2)

with colC:
    fig_animals = px.histogram(same_cluster_df, x="fav_animals", color="fav_animals")
    fig_animals.update_layout(title="🐾 Ulubione zwierzęta")
    st.plotly_chart(fig_animals, use_container_width=True)

with colD:
    # Konwersja na stringi, brak problemów z Categorical
    same_cluster_df["fav_place"] = same_cluster_df["fav_place"].astype(str).fillna("Brak miejsca")
    same_cluster_df["fav_animals"] = same_cluster_df["fav_animals"].astype(str).fillna("Brak ulubionych")

    fig_place = px.sunburst(
        same_cluster_df,
        path=["fav_place", "fav_animals"],
        title="🌞 Sunburst zainteresowań"
    )
    st.plotly_chart(fig_place, use_container_width=True)

colE, colF = st.columns(2)
with colE:
    fig_gender = px.pie(same_cluster_df, names="gender", title="🚻 Płeć w grupie")
    st.plotly_chart(fig_gender, use_container_width=True)

# ------------------------------
# RADAR / POLAR CHART
# ------------------------------
st.subheader("🧭 Twoje cechy vs średnia grupy")

def encode(df):
    df = df.copy()
    # Fill NA przed mapowaniem
    df["age"] = df["age"].fillna("unknown")
    df["edu_level"] = df["edu_level"].fillna("Podstawowe")
    df["fav_animals"] = df["fav_animals"].fillna("Brak ulubionych")
    df["fav_place"] = df["fav_place"].fillna("Inne")
    df["gender"] = df["gender"].fillna("Mężczyzna")

    mapping_age = {'<18':1, '18-24':2, '25-34':3, '35-44':4, '45-54':5, '55-64':6, '>=65':7, 'unknown':0}
    df["age_enc"] = df["age"].map(mapping_age).astype(int)
    df["edu_enc"] = df["edu_level"].map({'Podstawowe':1, 'Średnie':2, 'Wyższe':3}).astype(int)
    df["animal_enc"] = df["fav_animals"].map({"Brak ulubionych":1, "Psy":2, "Koty":3, "Inne":4, "Koty i Psy":5}).astype(int)
    df["place_enc"] = df["fav_place"].map({"Nad wodą":1, "W lesie":2, "W górach":3, "Inne":4}).astype(int)
    df["gender_enc"] = df["gender"].map({"Mężczyzna":1, "Kobieta":2}).astype(int)

    return df

encoded_all = encode(all_df)
encoded_user = encode(person_df)

cluster_mean = encoded_all[encoded_all["Cluster"] == predicted_cluster_id][
    ["age_enc", "edu_enc", "animal_enc", "place_enc", "gender_enc"]
].mean()

labels = ["Wiek", "Wykształcenie", "Zwierzęta", "Miejsce", "Płeć"]

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r = encoded_user[["age_enc","edu_enc","animal_enc","place_enc","gender_enc"]].values.flatten(),
    theta = labels,
    fill='toself',
    name='Ty 💛'
))

fig_radar.add_trace(go.Scatterpolar(
    r = cluster_mean.values,
    theta = labels,
    fill='toself',
    name='Średnia grupy 💙'
))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=True,
    title="📌 Porównanie cech"
)

st.plotly_chart(fig_radar, use_container_width=True)

# ------------------------------
# TABLE WITH CLUSTER MEMBERS
# ------------------------------
st.header("📋 Lista osób z Twojej grupy")
st.dataframe(same_cluster_df, use_container_width=True)
