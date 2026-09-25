# uv run streamlit run app.py

import streamlit as st
import pandas as pd
import joblib
import os


CHEMIN_PIPELINE = "model/pipeline_final.joblib"
CHEMIN_COLONNES = "model/colonnes_attendues.joblib"


if not os.path.exists(CHEMIN_PIPELINE) or not os.path.exists(CHEMIN_COLONNES):
    st.error(
        "Fichiers du modèle introuvables. "
        "Exécute d'abord notebook_analyse.py pour générer "
        "model/pipeline_final.joblib et model/colonnes_attendues.joblib."
    )
    st.stop()

pipeline = joblib.load(CHEMIN_PIPELINE)
colonnes_attendues = joblib.load(CHEMIN_COLONNES)

st.title("Prédiction du score d'examen d'un étudiant")
st.write(
    "Renseigne le profil de l'étudiant ci-dessous pour estimer son score "
    "final à l'examen (Exam_Score)."
)


st.sidebar.header("Profil de l'étudiant")


hours_studied = st.sidebar.slider("Heures d'étude par semaine", 0, 40, 20)
attendance = st.sidebar.slider("Taux de présence (%)", 0, 100, 80)
sleep_hours = st.sidebar.slider("Heures de sommeil par nuit", 0, 12, 7)
previous_scores = st.sidebar.slider("Score à l'examen précédent", 0, 100, 70)
tutoring_sessions = st.sidebar.slider("Séances de tutorat / mois", 0, 10, 2)
physical_activity = st.sidebar.slider("Activité physique (h/semaine)", 0, 20, 3)


gender = st.sidebar.selectbox("Genre", ["Male", "Female"])
school_type = st.sidebar.selectbox("Type d'établissement", ["Public", "Private"])
extracurricular = st.sidebar.selectbox("Activités extrascolaires", ["Yes", "No"])
internet_access = st.sidebar.selectbox("Accès à internet", ["Yes", "No"])
learning_disabilities = st.sidebar.selectbox("Troubles d'apprentissage", ["Yes", "No"])
peer_influence = st.sidebar.selectbox("Influence des pairs", ["Positive", "Neutral", "Negative"])


parental_involvement = st.sidebar.selectbox("Implication parentale", ["Low", "Medium", "High"])
access_to_resources = st.sidebar.selectbox("Accès aux ressources", ["Low", "Medium", "High"])
motivation_level = st.sidebar.selectbox("Niveau de motivation", ["Low", "Medium", "High"])
family_income = st.sidebar.selectbox("Revenu familial", ["Low", "Medium", "High"])
teacher_quality = st.sidebar.selectbox("Qualité des enseignants", ["Low", "Medium", "High"])
parental_education = st.sidebar.selectbox(
    "Niveau d'éducation des parents", ["High School", "College", "Postgraduate"]
)
distance_from_home = st.sidebar.selectbox("Distance domicile-école", ["Near", "Moderate", "Far"])

bouton_predire = st.sidebar.button("Prédire le score")


def construire_ligne_encodee():

    # a) On part d'un dictionnaire avec les 6 variables numériques
    ligne = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Tutoring_Sessions": tutoring_sessions,
        "Physical_Activity": physical_activity,
    }

    # b) Encodage ordinal : mêmes dictionnaires que dans le notebook
    mapping_low_medium_high = {"Low": 0, "Medium": 1, "High": 2}
    mapping_education = {"High School": 0, "College": 1, "Postgraduate": 2}
    mapping_distance = {"Near": 0, "Moderate": 1, "Far": 2}

    ligne["Parental_Involvement"] = mapping_low_medium_high[parental_involvement]
    ligne["Access_to_Resources"] = mapping_low_medium_high[access_to_resources]
    ligne["Motivation_Level"] = mapping_low_medium_high[motivation_level]
    ligne["Family_Income"] = mapping_low_medium_high[family_income]
    ligne["Teacher_Quality"] = mapping_low_medium_high[teacher_quality]
    ligne["Parental_Education_Level"] = mapping_education[parental_education]
    ligne["Distance_from_Home"] = mapping_distance[distance_from_home]

    # c) One-hot encoding manuel des variables nominales.

    ligne[f"Gender_{gender}"] = 1
    ligne[f"School_Type_{school_type}"] = 1
    ligne[f"Extracurricular_Activities_{extracurricular}"] = 1
    ligne[f"Internet_Access_{internet_access}"] = 1
    ligne[f"Learning_Disabilities_{learning_disabilities}"] = 1
    ligne[f"Peer_Influence_{peer_influence}"] = 1

    # d) Transformation en DataFrame à une seule ligne
    df_ligne = pd.DataFrame([ligne])

    # e) reindex(columns=colonnes_attendues, fill_value=0) :

    df_ligne = df_ligne.reindex(columns=colonnes_attendues, fill_value=0)

    return df_ligne



if bouton_predire:
    try:
        X_utilisateur = construire_ligne_encodee()


        if X_utilisateur.isnull().values.any():
            st.error("Une ou plusieurs valeurs saisies sont invalides.")
        else:
            score_predit = pipeline.predict(X_utilisateur)[0]

            score_predit = max(0, min(100, score_predit))

            st.subheader("Résultat de la prédiction")
            st.metric("Score d'examen estimé", f"{score_predit:.1f} / 100")


            SEUIL_RISQUE = 60
            if score_predit < SEUIL_RISQUE:
                st.warning(
                    f"⚠️ Étudiant à risque d'échec (score estimé < {SEUIL_RISQUE}). "
                    "Un accompagnement pédagogique est recommandé."
                )
            else:
                st.success("✅ Étudiant sur une trajectoire de réussite.")

    except Exception as erreur:

        st.error(f"Une erreur est survenue lors de la prédiction : {erreur}")