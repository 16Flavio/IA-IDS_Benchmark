# 🛡️ IDS-Benchmark: Comparatif ML/DL pour la Cybersécurité

Comparaison des performances des architectures de détection d'intrusions (IDS) sur des datasets historiques et modernes.

## 📊 Données & Téléchargements
Les datasets étant volumineux, ils ne sont pas inclus dans le dépôt git.
* **Dataset Actuel du Benchmark** : `CIC_IDS2017`

### 📥 Comment obtenir les données ?
1.  Créez un dossier `data/` à la racine du projet.
2.  Téléchargez les fichiers CSV depuis les sources officielles :
    * **NSL-KDD** : [Télécharger ici (UNB)](https://www.kaggle.com/datasets/hassan06/nslkdd)
    * **CIC-IDS2017** : [Télécharger ici (UNB)](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
3.  Placez les fichiers (ex: `KDDTrain+.txt` ou `Wednesday-workingHours.pcap_ISCX.csv`) dans les sous-dossiers correspondants (`data/nsl_kdd/` ou `data/cic_ids2017/`).

### 🧠 Modèles Pré-entraînés
Les modèles entraînés (`.joblib` et `.keras`) sont disponibles dans la section **[Releases](../../releases)** de ce dépôt GitHub pour éviter de tout ré-entraîner.

## 🚀 Installation & Usage
```bash
git clone [https://github.com/16Flavio/AI-IDS-Benchmark.git](https://github.com/16Flavio/AI-IDS-Benchmark.git)
pip install -r requirements.txt

# A. Entraîner les modèles (Si vous avez téléchargé les données)
python main.py --mode train --dataset cic_ids2017

# B. Évaluer et générer ce rapport
python main.py --mode eval --dataset cic_ids2017

# C. Simulation Temps Réel (Dashboard SOC)
python main.py --mode sim --dataset cic_ids2017
```

## 📈 Résultats de l'Évaluation (14/12/2025 à 00:45)

### Performance Globale
| Modèle | Précision | F1-Score | Temps (s) |
| :--- | :--- | :--- | :--- |
| **Random Forest** | 99.89% | 0.9972 | 3.3721 |
| **Deep Learning** | 98.43% | 0.9604 | 3.4116 |
| **Hybride** | 98.43% | 0.9605 | 33.2860 |
| **Traditionnel (Règles)** | 80.12% | 0.0004 | 28.3296 |
| **Auto-Encoder (Zero-Day)** | 77.87% | 0.5162 | 40.7939 |
| **XGBoost** | 99.92% | 0.9980 | 0.3769 |


### 🔍 Matrices de Confusion
| Random Forest | Deep Learning |
| :---: | :---: |
| ![RF](results/Random_Forest_cm.png) | ![DL](results/Deep_Learning_cm.png) |

| XGBoost | Hybride (IA + Règles) |
| :---: | :---: |
| ![XGB](results/XGBoost_cm.png) | ![Hybrid](results/Hybride_cm.png) |

### 📉 Courbes ROC
| Random Forest | Deep Learning |
| :---: | :---: |
| ![RF ROC](results/Random_Forest_roc.png) | ![DL ROC](results/Deep_Learning_roc.png) |
