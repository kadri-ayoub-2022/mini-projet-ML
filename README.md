# Mini-Projet Machine Learning
Ce mini-projet est une application web interactive développée avec Streamlit, permettant de tester des modèles de Machine Learning supervisés et non supervisés.

# 📋 Prérequis
Python 3.8 ou supérieur

Pip (gestionnaire de paquets Python)

# 🚀 Étapes pour exécuter le projet

Étape 1 : Installer les dépendances
bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn
Étape 3 : Lancer l'application supervisée
Ouvrir un terminal et se placer dans le dossier supervised :

bash
cd supervised
Lancer l'application Streamlit :

bash
streamlit run app.py
Accéder à l'application dans votre navigateur :

text
http://localhost:8501
Étape 4 : Lancer l'application non supervisée
Ouvrir un nouveau terminal (pour garder les deux applications en parallèle)

Se placer dans le dossier unsupervised :

bash
cd unsupervised
Lancer l'application Streamlit sur un port différent :

bash
streamlit run app.py --server.port 8502
Accéder à l'application dans votre navigateur :

text
http://localhost:8502