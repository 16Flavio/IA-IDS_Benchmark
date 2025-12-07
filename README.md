# 🛡️ IDS-Benchmark: Comparatif ML/DL pour la Cybersécurité

Comparaison des performances des architectures de détection d'intrusions (IDS) sur des datasets historiques et modernes.

## 📊 Datasets Étudiés
* **NSL-KDD** : Dataset académique de référence.
* **CIC-IDS2017** : Dataset moderne incluant des attaques DDoS, Brute Force et Botnet.

## 🧠 Méthodes Comparées
1.  **Approche Traditionnelle** : Détection basée sur des règles et seuils statistiques.
2.  **Machine Learning** : Random Forest (Supervisé).
3.  **Deep Learning** : Multi-Layer Perceptron (MLP).
4.  **Approche Hybride** : Deep Learning assisté par logique de décision (Neuro-Symbolique simplifié) pour réduire les faux positifs.

## 🚀 Installation & Usage

```bash
git clone https://github.com/16Flavio/AI-IDS-Benchmark.git
pip install -r requirements.txt
python main.py
```

## 📈 Résultats Préliminaires

| Modèle | Précision | F1-Score | Remarque | 

| Check | --- | --- | --- | 

| Random Forest | 99.94% | 0.99 | Très rapide et robuste | 

| Hybride | 66.29% | 0.14 | Meilleure gestion des cas limites | 

| Deep Learning | 84.37% | 0.82 | Nécessite plus de données et une recherche de meilleurs hyperparametres | 