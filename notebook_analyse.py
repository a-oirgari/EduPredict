import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib


df = pd.read_csv("data/StudentPerformanceFactors.csv")

print(type(df))
print(df.shape)
df.dtypes


# ## Feature Story 1 - Tâche 2 : Analyse exploratoire (EDA)


df.info()
df.head()


# --- Analyse descriptive ---

df.describe()


colonnes_categorielles = df.select_dtypes(include="object").columns
for col in colonnes_categorielles:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())

# --- Valeurs manquantes ---
print(df.isnull().sum())

# --- Doublons ---
print("Nombre de doublons :", df.duplicated().sum())

# --- Distributions ---
# Histogramme de la variable cible Exam_Score
plt.figure(figsize=(8, 5))
sns.histplot(df["Exam_Score"], kde=True)
plt.title("Distribution du score d'examen (Exam_Score)")
plt.xlabel("Exam_Score")
plt.show()

# Histogramme de Hours_Studied
plt.figure(figsize=(8, 5))
sns.histplot(df["Hours_Studied"], kde=True)
plt.title("Distribution des heures d'étude (Hours_Studied)")
plt.xlabel("Hours_Studied")
plt.show()

# --- Matrice de corrélation (avant encodage) ---

plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation (variables numériques uniquement)")
plt.show()


# ## Feature Story 1 - Tâche 3 : Prétraitement

# --- 3.1 Imputation des valeurs manquantes par le mode ---
colonnes_a_imputer = ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]

imputer = SimpleImputer(strategy="most_frequent")
df[colonnes_a_imputer] = imputer.fit_transform(df[colonnes_a_imputer])

print(df[colonnes_a_imputer].isnull().sum())

# --- 3.2 Suppression des doublons ---
df.drop_duplicates(inplace=True)
print("Nouvelle taille après suppression des doublons :", df.shape)

# --- 3.3 Valeurs aberrantes (outliers) sur Exam_Score ---
# a) Visualisation avec un boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["Exam_Score"])
plt.title("Boxplot de Exam_Score (avant nettoyage des outliers)")
plt.show()

# b) Méthode IQR
Q1 = df["Exam_Score"].quantile(0.25)
Q3 = df["Exam_Score"].quantile(0.75)
IQR = Q3 - Q1
borne_basse = Q1 - 1.5 * IQR
borne_haute = Q3 + 1.5 * IQR
print(f"Bornes IQR : [{borne_basse:.2f} ; {borne_haute:.2f}]")

# c) Vérification avec le z-score (nombre d'écarts-types par rapport à la moyenne)
z_scores = (df["Exam_Score"] - df["Exam_Score"].mean()) / df["Exam_Score"].std()
print("Nombre de lignes avec |z-score| > 3 :", (z_scores.abs() > 3).sum())

# d) Suppression selon les bornes IQR (méthode retenue par le brief)
df = df[(df["Exam_Score"] >= borne_basse) & (df["Exam_Score"] <= borne_haute)]
print("Taille après suppression des outliers :", df.shape)

# --- 3.4 One-hot encoding des variables NOMINALES (sans ordre) ---
colonnes_nominales = [
    "Gender",
    "School_Type",
    "Extracurricular_Activities",
    "Internet_Access",
    "Learning_Disabilities",
    "Peer_Influence",
]
df = pd.get_dummies(df, columns=colonnes_nominales)

# --- 3.5 Encodage ORDINAL des variables ordinales (avec ordre logique) ---
mapping_low_medium_high = {"Low": 0, "Medium": 1, "High": 2}
mapping_education = {"High School": 0, "College": 1, "Postgraduate": 2}
mapping_distance = {"Near": 0, "Moderate": 1, "Far": 2}

colonnes_ordinales_std = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Motivation_Level",
    "Family_Income",
    "Teacher_Quality",
]
for col in colonnes_ordinales_std:
    df[col] = df[col].map(mapping_low_medium_high)

df["Parental_Education_Level"] = df["Parental_Education_Level"].map(mapping_education)
df["Distance_from_Home"] = df["Distance_from_Home"].map(mapping_distance)

print(df.dtypes)

# --- Matrice de corrélation refaite APRÈS encodage ---
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Matrice de corrélation complète (après encodage)")
plt.show()

df.to_csv(
    "data/StudentPerformanceFactors_cleaned.csv",
    index=False
)
print("Dataset nettoyé sauvegardé avec succès.")

# --- 3.6 Séparation X (features) / y (cible) ---
X = df.drop(columns=["Exam_Score"])
y = df["Exam_Score"]

colonnes_attendues = list(X.columns)
print("Nombre de colonnes finales :", len(colonnes_attendues))

# --- 3.7 Découpage train / test (80% / 20%) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Taille train :", X_train.shape, " | Taille test :", X_test.shape)

# --- 3.8 StandardScaler sur les variables numériques uniquement ---

colonnes_numeriques = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]

preprocesseur = ColumnTransformer(
    transformers=[("scaler", StandardScaler(), colonnes_numeriques)],
    remainder="passthrough",
)



# ## Feature Story 2 : Entraînement des modèles (paramètres par défaut)

modeles = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    # "XGBRegressor": XGBRegressor(random_state=42),
    "SVR": SVR(),
}


resultats_defaut = {}

for nom, modele in modeles.items():

    pipeline = Pipeline(steps=[
        ("preprocesseur", preprocesseur),
        ("modele", modele),
    ])


    pipeline.fit(X_train, y_train)


    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    resultats_defaut[nom] = {"RMSE": rmse, "MAE": mae, "R2": r2, "pipeline": pipeline}
    print(f"{nom:25s} | RMSE={rmse:.3f} | MAE={mae:.3f} | R2={r2:.3f}")

# Tableau récapitulatif des résultats "par défaut"
df_resultats_defaut = pd.DataFrame({
    nom: {"RMSE": v["RMSE"], "MAE": v["MAE"], "R2": v["R2"]}
    for nom, v in resultats_defaut.items()
}).T
print(df_resultats_defaut)


# ## Feature Story 3 : Tuning des hyperparamètres (GridSearchCV)
grilles = {
    "LinearRegression": {
        "modele__fit_intercept": [True, False],
        "modele__positive": [True, False],
    },
    "RandomForestRegressor": {
        "modele__n_estimators": [100, 200, 300],
        "modele__max_depth": [None, 5, 10, 20],
        "modele__min_samples_split": [2, 5, 10],
    },
    # "XGBRegressor": {
    #     "modele__n_estimators": [100, 200, 300],
    #     "modele__learning_rate": [0.01, 0.05, 0.1],
    #     "modele__max_depth": [3, 5, 7],
    #     "modele__subsample": [0.7, 0.85, 1.0],
    # },
    "SVR": {
        "modele__C": [0.1, 1, 10],
        "modele__kernel": ["linear", "rbf"],
        "modele__epsilon": [0.01, 0.1, 0.5],
    },
}

meilleurs_pipelines = {}
resultats_tuning = {}

for nom, modele in modeles.items():
    pipeline = Pipeline(steps=[
        ("preprocesseur", preprocesseur),
        ("modele", modele),
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grilles[nom],
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    meilleur_pipeline = grid_search.best_estimator_
    meilleurs_pipelines[nom] = meilleur_pipeline

    y_pred = meilleur_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    resultats_tuning[nom] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"{nom:25s} | meilleurs params : {grid_search.best_params_}")
    print(f"{'':25s} | RMSE={rmse:.3f} | MAE={mae:.3f} | R2={r2:.3f}\n")

# Comparaison avant / après tuning
df_resultats_tuning = pd.DataFrame(resultats_tuning).T
print("=== AVANT tuning ===")
print(df_resultats_defaut)
print("\n=== APRÈS tuning ===")
print(df_resultats_tuning)


# ## Feature Story 4 : Évaluation et comparaison des modèles

# --- Scatter plots : prédictions vs valeurs réelles ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (nom, pipeline) in enumerate(meilleurs_pipelines.items()):
    y_pred = pipeline.predict(X_test)
    axes[i].scatter(y_test, y_pred, alpha=0.4)
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    axes[i].set_title(nom)
    axes[i].set_xlabel("Valeurs réelles")
    axes[i].set_ylabel("Valeurs prédites")

plt.tight_layout()
plt.show()

# --- Distribution des résidus (erreur = réel - prédit) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

residus_par_modele = {}
for i, (nom, pipeline) in enumerate(meilleurs_pipelines.items()):
    y_pred = pipeline.predict(X_test)
    residus = y_test - y_pred
    residus_par_modele[nom] = residus
    sns.histplot(residus, kde=True, ax=axes[i])
    axes[i].set_title(f"Résidus - {nom}")
    axes[i].set_xlabel("Résidu (réel - prédit)")

plt.tight_layout()
plt.show()

# --- Tableau récapitulatif final ---
print(df_resultats_tuning)

# --- Robustesse : écart-type des résidus + écart-type du R² en CV ---
from sklearn.model_selection import cross_val_score

robustesse = {}
for nom, pipeline in meilleurs_pipelines.items():
    ecart_type_residus = residus_par_modele[nom].std()

    scores_cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")

    robustesse[nom] = {
        "Ecart_type_residus": ecart_type_residus,
        "Ecart_type_R2_CV": scores_cv.std(),
        "Moyenne_R2_CV": scores_cv.mean(),
    }

df_robustesse = pd.DataFrame(robustesse).T
print(df_robustesse)

# --- Choix du modèle final ---
NOM_MODELE_FINAL = "SVR"

pipeline_final = meilleurs_pipelines[NOM_MODELE_FINAL]
print(f"Modèle final retenu : {NOM_MODELE_FINAL}")


# ## Feature Story 5 (partie 1) : Export du pipeline final

os.makedirs("model", exist_ok=True)
# On sauvegarde :
joblib.dump(pipeline_final, "model/pipeline_final.joblib")
joblib.dump(colonnes_attendues, "model/colonnes_attendues.joblib")

print("Pipeline et colonnes exportés dans le dossier model/")