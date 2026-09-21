# Dictionnaire des données

Source : [Student Performance Factors (Kaggle)](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

Le jeu de données contient **19 variables explicatives** et **1 variable cible** (`Exam_Score`).

## Variables numériques

| Colonne | Description |
|---|---|
| `Hours_Studied` | Heures d'étude hebdomadaires |
| `Attendance` | Taux de présence en cours (en %) |
| `Sleep_Hours` | Heures de sommeil moyennes par nuit |
| `Previous_Scores` | Scores obtenus aux examens précédents |
| `Tutoring_Sessions` | Nombre de séances de tutorat suivies par mois |
| `Physical_Activity` | Activité physique hebdomadaire (en heures) |

## Variables catégorielles nominales (sans ordre)

Encodage utilisé : **one-hot encoding**.

| Colonne | Description | Valeurs |
|---|---|---|
| `Gender` | Genre | Male / Female |
| `School_Type` | Type d'établissement | Public / Private |
| `Extracurricular_Activities` | Pratique d'activités extrascolaires | Yes / No |
| `Internet_Access` | Accès à internet | Yes / No |
| `Learning_Disabilities` | Présence de troubles d'apprentissage | Yes / No |
| `Peer_Influence` | Influence des pairs | Positive / Neutral / Negative |

## Variables catégorielles ordinales (avec ordre)

Encodage utilisé : **encodage ordinal** (0, 1, 2 selon l'ordre croissant).

| Colonne | Description | Ordre des valeurs |
|---|---|---|
| `Parental_Involvement` | Implication parentale | Low < Medium < High |
| `Access_to_Resources` | Accès aux ressources pédagogiques | Low < Medium < High |
| `Motivation_Level` | Niveau de motivation | Low < Medium < High |
| `Family_Income` | Revenu familial | Low < Medium < High |
| `Teacher_Quality` | Qualité perçue des enseignants (valeurs manquantes) | Low < Medium < High |
| `Parental_Education_Level` | Niveau d'éducation des parents (valeurs manquantes) | High School < College < Postgraduate |
| `Distance_from_Home` | Distance domicile-établissement (valeurs manquantes) | Near < Moderate < Far |

## Variable cible

| Colonne | Description |
|---|---|
| `Exam_Score` | Score final à l'examen (valeur continue) |

## Valeurs manquantes

Les colonnes `Teacher_Quality`, `Parental_Education_Level` et `Distance_from_Home` contiennent des valeurs manquantes. Elles sont remplacées par le **mode** (valeur la plus fréquente) de chaque colonne.
