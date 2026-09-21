# Prédiction du score d'examen des étudiants

Projet de régression permettant de prédire le score final d'examen
(`Exam_Score`) d'un étudiant, et d'identifier en amont les étudiants à
risque d'échec, afin d'aider l'équipe pédagogique dans son accompagnement.

Jeu de données : (https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

## Structure des fichiers

```
projet_score_examen/
├── data/
│   └── StudentPerformanceFactors.csv   # à télécharger depuis Kaggle 
├── model/
│   ├── pipeline_final.joblib           # pipeline complet (scaler + modèle) - généré par le notebook
│   └── colonnes_attendues.joblib       # liste ordonnée des colonnes attendues - généré par le notebook
├── notebook_analyse.py                 # EDA, prétraitement, entraînement, tuning, évaluation
├── app.py                              # interface Streamlit
├── dictionnaire_des_donnees.md         # description des colonnes du dataset
└── README.md
```

## Instructions d'exécution

### 1. Installer les dépendances

```bash
uv add pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit
```

### 2. Récupérer les données

Télécharger le dataset depuis Kaggle et le placer dans :
`data/StudentPerformanceFactors.csv`

### 3. Exécuter l'analyse et entraîner le modèle

Ouvrir `notebook_analyse.py` dans VSCode (cellules `# %%`) ou copier son
contenu dans un notebook Jupyter, puis exécuter toutes les cellules dans
l'ordre. À la fin de l'exécution, deux fichiers sont générés dans `model/` :
- `pipeline_final.joblib` : le pipeline complet (StandardScaler + modèle
  de régression retenu)
- `colonnes_attendues.joblib` : la liste ordonnée des colonnes utilisées
  à l'entraînement (nécessaire pour que l'app fasse les mêmes prédictions)

### 4. Lancer l'application

```bash
streamlit run app.py
```


## Description du projet

Le projet suit 6 étapes (Feature Stories) :

1. **Analyse et préparation des données** : chargement, EDA (structure,
   statistiques descriptives, valeurs manquantes, doublons, distributions,
   corrélations), puis prétraitement (imputation par le mode, suppression
   des doublons et des outliers via la méthode IQR, one-hot encoding des
   variables nominales, encodage ordinal des variables ordinales,
   découpage train/test 80/20, standardisation des variables numériques).
2. **Entraînement des modèles** : `LinearRegression`, `RandomForestRegressor`,
   `XGBRegressor`, `SVR`, chacun encapsulé dans un pipeline Scikit-learn,
   évalués avec RMSE, MAE et R².
3. **Tuning des hyperparamètres** : optimisation via `GridSearchCV`
   (validation croisée à 3 plis, métrique R²) pour chaque modèle.
4. **Évaluation et comparaison** : scatter plots prédictions vs réel,
   distribution des résidus, tableau récapitulatif, analyse de la
   robustesse (écart-type des résidus et du R² en validation croisée),
   choix du modèle final.
5. **Test et déploiement** : export du pipeline complet avec `joblib`,
   application Streamlit permettant de saisir le profil d'un étudiant et
   d'obtenir son score estimé, avec signalement des étudiants à risque.
6. **Documentation et reproductibilité** : commentaires détaillés dans le
   notebook, ce README, et planification des tâches dans Jira (Epics =
   Feature Stories, tickets = tâches).

## Variable cible

`Exam_Score` : score final obtenu par l'étudiant à l'examen.

Voir `dictionnaire_des_donnees.md` pour le détail de toutes les colonnes.