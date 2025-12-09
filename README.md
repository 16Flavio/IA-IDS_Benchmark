# 🛡️ IDS-Benchmark: Comparatif ML/DL pour la Cybersécurité

Comparaison des performances des architectures de détection d'intrusions (IDS) sur des datasets historiques et modernes.

## 📊 Datasets Étudiés
* **NSL-KDD** : Dataset académique de référence.
* **CIC-IDS2017** : Dataset moderne incluant des attaques DDoS, Brute Force et Botnet.

## 🧠 Méthodes Comparées
1.  **Approche Traditionnelle** : Détection basée sur règles (Rule-based).
2.  **Machine Learning** : Random Forest (Supervisé).
3.  **Deep Learning** : Multi-Layer Perceptron (MLP) avec BatchNormalization.
4.  **Approche Hybride** : Deep Learning assisté par logique experte (Neuro-Symbolique).

## 🚀 Installation & Usage
```bash
git clone https://github.com/16Flavio/AI-IDS-Benchmark.git
pip install -r requirements.txt

# 1. Entraîner les modèles (Sauvegarde dans /models)
python main.py --mode train --dataset cic_ids2017

# 2. Évaluer et générer les graphiques/README (Lecture depuis /models)
python main.py --mode eval --dataset cic_ids2017
```

## 📈 Résultats (Dernière mise à jour : 09/12/2025 à 11:03)
| Modèle | Précision | F1-Score | Temps (s) |
| :--- | :--- | :--- | :--- |
| **Random Forest** | 99.89% | 0.9973 | 5.4984 |
| **Deep Learning** | 99.47% | 0.9865 | 1.8857 |
| **Hybride** | 99.26% | 0.9809 | 2.7802 |
| **Traditionnel** | 80.32% | 0.0000 | 13.3568 |


## 🔍 Analyse
* Consultez le dossier `/results` pour visualiser les **Matrices de Confusion** et les **Courbes ROC**.
* Consultez `notebooks/01_Data_Exploration.ipynb` pour l'analyse exploratoire des données.
