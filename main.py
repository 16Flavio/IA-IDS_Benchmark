import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=Tout, 1=Info, 2=Warning, 3=Error

import pandas as pd
from src.data_loader import DataLoader
from src.methods import RuleBasedDetector, MLDetector, DLDetector, HybridDetector
from sklearn.metrics import accuracy_score, f1_score
import time

# CHOIX DU DATASET
DATASET = 'cic_ids2017' # ou 'nsl_kdd'

print(f"--- BENCHMARK CYBERSÉCURITÉ : {DATASET.upper()} ---")

# Chargement
loader = DataLoader(DATASET)
X_train, X_test, y_train, y_test = loader.load_data()
print(f"Données chargées. Train: {X_train.shape}, Test: {X_test.shape}")

# Réservoir de résultats
results = []

# Exécution des modèles
models = {
    "Traditionnel (Règles)": RuleBasedDetector(),
    "Machine Learning (RF)": MLDetector(),
    "Deep Learning (MLP)": DLDetector(X_train.shape[1]),
    # Hybride sera instancié après le DL
}

# Boucle d'évaluation
dl_instance = None # Pour garder le modèle DL entrainé pour l'hybride

for name, model in models.items():
    print(f"Traitement : {name}...")
    start = time.time()
    
    # Entrainement (sauf pour règles)
    if name != "Traditionnel (Règles)":
        model.train(X_train, y_train)
    
    if name == "Deep Learning (MLP)":
        dl_instance = model
        
    # Prédiction
    preds = model.predict(X_test)
    if name == "Deep Learning (MLP)": # Flatten pour le DL
        preds = preds.flatten()
        
    duration = time.time() - start
    
    # Métriques
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    results.append({
        "Modèle": name,
        "Précision": acc,
        "F1-Score": f1,
        "Temps (s)": duration
    })

# Ajout du modèle Hybride (DL + Logique)
print("Traitement : Hybride (DL + Règles)...")
hybrid = HybridDetector(dl_instance)
h_preds = hybrid.predict(X_test)
results.append({
    "Modèle": "Hybride (DL + Logique)",
    "Précision": accuracy_score(y_test, h_preds),
    "F1-Score": f1_score(y_test, h_preds),
    "Temps (s)": "N/A (Inférence)"
})

# Affichage final
res_df = pd.DataFrame(results)
print("\n--- RÉSULTATS COMPARATIFS ---")
print(res_df.sort_values(by="F1-Score", ascending=False))