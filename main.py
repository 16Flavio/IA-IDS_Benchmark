import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import pandas as pd
import time
import sys
from src.data_loader import DataLoader
from src.methods import RuleBasedDetector, MLDetector, DLDetector, HybridDetector, XGBoostDetector, HAS_XGBOOST, AnomalyDetector
from src.evaluation import Evaluator
from src.simulation import TrafficSimulator
# Import optionnel pour l'attaque (si fichier présent)
try:
    from src.adversarial import test_robustness
    HAS_ADVERSARIAL = True
except ImportError:
    HAS_ADVERSARIAL = False

from sklearn.metrics import accuracy_score, f1_score
import datetime

def update_readme(results_df, dataset_name):
    """
    Met à jour le fichier README.md avec :
    1. Les résultats (Tableau)
    2. Les images (Matrices/ROC)
    3. Les liens de téléchargement (Datasets/Modèles)
    """
    # 1. Tableau des scores
    md_table = "| Modèle | Précision | F1-Score | Temps (s) |\n"
    md_table += "| :--- | :--- | :--- | :--- |\n"
    
    for _, row in results_df.iterrows():
        t = row['Temps (s)']
        temps_str = f"{t:.4f}" if isinstance(t, float) else str(t)
        md_table += f"| **{row['Modèle']}** | {row['Précision']:.2%} | {row['F1-Score']:.4f} | {temps_str} |\n"

    date_now = datetime.datetime.now().strftime("%d/%m/%Y à %H:%M")
    
    # 2. Liens vers les datasets (Statiques)
    link_nsl = "https://www.kaggle.com/datasets/hassan06/nslkdd"
    link_cic = "https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset"
    
    # 3. Contenu complet du README
    readme_content = f"""# 🛡️ IDS-Benchmark: Comparatif ML/DL pour la Cybersécurité

Comparaison des performances des architectures de détection d'intrusions (IDS) sur des datasets historiques et modernes.

## 📊 Données & Téléchargements
Les datasets étant volumineux, ils ne sont pas inclus dans le dépôt git.
* **Dataset Actuel du Benchmark** : `{dataset_name.upper()}`

### 📥 Comment obtenir les données ?
1.  Créez un dossier `data/` à la racine du projet.
2.  Téléchargez les fichiers CSV depuis les sources officielles :
    * **NSL-KDD** : [Télécharger ici (UNB)]({link_nsl})
    * **CIC-IDS2017** : [Télécharger ici (UNB)]({link_cic})
3.  Placez les fichiers (ex: `KDDTrain+.txt` ou `Wednesday-workingHours.pcap_ISCX.csv`) dans les sous-dossiers correspondants (`data/nsl_kdd/` ou `data/cic_ids2017/`).

### 🧠 Modèles Pré-entraînés
Les modèles entraînés (`.joblib` et `.keras`) sont disponibles dans la section **[Releases](../../releases)** de ce dépôt GitHub pour éviter de tout ré-entraîner.

## 🚀 Installation & Usage
```bash
git clone [https://github.com/16Flavio/AI-IDS-Benchmark.git](https://github.com/16Flavio/AI-IDS-Benchmark.git)
pip install -r requirements.txt

# A. Entraîner les modèles (Si vous avez téléchargé les données)
python main.py --mode train --dataset {dataset_name}

# B. Évaluer et générer ce rapport
python main.py --mode eval --dataset {dataset_name}

# C. Simulation Temps Réel (Dashboard SOC)
python main.py --mode sim --dataset {dataset_name}
```

## 📈 Résultats de l'Évaluation ({date_now})

### Performance Globale
{md_table}

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
"""
    try:
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("\n[AUTO] README.md mis à jour avec les liens de téléchargement !")
    except Exception as e:
        print(f"\n[ATTENTION] Erreur écriture README : {e}")

if __name__ == "__main__":
    # Configuration des arguments ligne de commande
    parser = argparse.ArgumentParser(description="AI IDS Benchmark")
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'sim', 'attack'], default='train', help="Mode: 'train', 'eval', 'sim' ou 'attack'")
    parser.add_argument('--dataset', type=str, default='cic_ids2017', help="Dataset: 'nsl_kdd' ou 'cic_ids2017'")
    args = parser.parse_args()

    # Dossier pour sauvegarder les modèles
    MODEL_DIR = "models"
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print(f"--- BENCHMARK CYBERSÉCURITÉ : {args.dataset.upper()} | MODE : {args.mode.upper()} ---")

    # Chargement des données
    loader = DataLoader(args.dataset)
    X_train, X_test, X_val, y_train, y_test, y_val, labels_test, labels_val = loader.load_data()

    # Initialisation des modèles
    rf_model = MLDetector()
    dl_model = DLDetector(input_shape=X_train.shape[1]) 
    rb_model = RuleBasedDetector()
    
    # Init Auto-Encoder (Zero-Day)
    ae_model = AnomalyDetector(input_shape=X_train.shape[1])

    if HAS_XGBOOST:
        xgb_model = XGBoostDetector()
    else:
        xgb_model = None

    # Chemins de sauvegarde
    rf_path = os.path.join(MODEL_DIR, f"rf_{args.dataset}.joblib")
    dl_path = os.path.join(MODEL_DIR, f"dl_{args.dataset}.keras")
    xgb_path = os.path.join(MODEL_DIR, f"xgb_{args.dataset}.joblib")
    rb_path = os.path.join(MODEL_DIR, f"rb_{args.dataset}.joblib")
    ae_path = os.path.join(MODEL_DIR, f"ae_{args.dataset}.keras") # Nom base

    # --- MODE ENTRAINEMENT ---
    if args.mode == 'train':
        print("\n[START] Entraînement des modèles...")
        
        # 1. Règles Stastistiques (Training léger)
        print(" -> Règle Based (Statistiques)...")
        rb_model.fit(X_train, y_train)
        rb_model.save_model(rb_path)
        
        # 2. Auto-Encoder (Zero-Day / Non-Supervisé)
        print(" -> Auto-Encoder (Zero-Day Detection)...")
        ae_model.fit(X_train, y_train)
        ae_model.save_model(ae_path)

        # Random Forest
        print(" -> Random Forest...")
        rf_model.train(X_train, y_train)
        rf_model.save_model(rf_path)
        
        # XGBoost
        if xgb_model:
            print(" -> XGBoost...")
            # On utilise le set de TEST (interne) pour le monitoring
            xgb_model.train(X_train, y_train, X_val=X_test, y_val=y_test)
            xgb_model.save_model(xgb_path)
        
        # Deep Learning
        print(" -> Deep Learning...")
        # On utilise le set de TEST (interne) pour le monitoring/early stopping
        dl_model.train(X_train, y_train, X_val=X_test, y_val=y_test)
        dl_model.save_model(dl_path)
        
        print("\nEntraînement terminé. Modèles sauvegardés dans /models")

    # --- MODE EVALUATION ---
    elif args.mode == 'eval':
        print("\n[START] Évaluation complète sur le set de VALIDATION (jamais vu)...")
        evaluator = Evaluator(output_dir='results')
        
        # Chargement des modèles
        print(" -> Chargement des modèles...")
        rf_model.load_model(rf_path)
        
        if xgb_model:
            xgb_model.load_model(xgb_path)
            
        rb_model.load_model(rb_path) # Important pour l'hybride
        
        dl_loaded = DLDetector(input_shape=None) 
        dl_loaded.load_model(dl_path)
        
        ae_loaded = AnomalyDetector(input_shape=None)
        ae_loaded.load_model(ae_path)
        
        # Modèle Hybride (Intègre DL + Règles Statistiques)
        hybrid_model = HybridDetector(dl_loaded, rb_model)
        
        models_to_eval = {
            "Random Forest": rf_model,
            "Deep Learning": dl_loaded,
            "Hybride": hybrid_model,
            "Traditionnel (Règles)": rb_model,
            "Auto-Encoder (Zero-Day)": ae_loaded
        }
        
        if xgb_model:
            models_to_eval["XGBoost"] = xgb_model
        
        results = []

        # ON UTILISE LE SET DE VALIDATION (X_val) ICI !
        X_target = X_val
        y_target = y_val

        for name, model in models_to_eval.items():
            print(f" -> Testing {name}...")
            start = time.time()
            
            preds = model.predict(X_target)
            if hasattr(preds, "flatten"): # Pour DL/Numpy array
                preds = preds.flatten()
            
            duration = time.time() - start
            
            # Métriques de base
            acc = accuracy_score(y_target, preds)
            f1 = f1_score(y_target, preds)
            
            results.append({"Modèle": name, "Précision": acc, "F1-Score": f1, "Temps (s)": duration})
            
            # Graphiques avancés via evaluation.py
            evaluator.plot_confusion_matrix(y_target, preds, name)
            evaluator.save_report(y_target, preds, name)
            
            # Courbes ROC (si le modèle supporte predict_proba)
            try:
                probs = model.predict_proba(X_target)
                if name == "Deep Learning":
                    probs = probs.flatten()
                elif hasattr(probs, "shape") and probs.shape[1] == 2:
                    probs = probs[:, 1]
                elif name == "Auto-Encoder (Zero-Day)":
                    probs = probs # Déjà en 1D (MSE)
                
                evaluator.plot_roc_curve(y_target, probs, name)
            except Exception as e:
                print(f"   (Pas de ROC pour {name}: {e})")

        # Affichage final console
        res_df = pd.DataFrame(results)
        print("\n--- RÉSULTATS FINAUX (VALIDATION SET) ---")
        print(res_df.sort_values(by="F1-Score", ascending=False))
        update_readme(res_df, args.dataset)
        print(f"\nGraphiques sauvegardés dans le dossier /results")
    
    elif args.mode == 'sim':
        print("\n[START] Simulateur Temps Réel (sur Validation Set)...")
        
        if not os.path.exists(rf_path):
            print(f"ERREUR: Modèles non trouvés.")
            sys.exit(1)

        rf_model.load_model(rf_path)
        rb_model.load_model(rb_path)
        
        dl_loaded = DLDetector(input_shape=None) 
        dl_loaded.load_model(dl_path)
        
        ae_loaded = AnomalyDetector(input_shape=None)
        ae_loaded.load_model(ae_path)

        hybrid_model = HybridDetector(dl_loaded, rb_model)
        
        if xgb_model: xgb_model.load_model(xgb_path)
        
        simulation_models = {
            "RandomForest": rf_model,
            "DeepLearn": dl_loaded,
            "Hybride": hybrid_model,
            "AutoEncoder": ae_loaded
        }
        if xgb_model: simulation_models["XGBoost"] = xgb_model
        
        # On passe 'labels_val' au simulateur car on utilise X_val
        sim = TrafficSimulator(simulation_models, X_val, y_val, labels_val)
        sim.run(num_packets=100, delay=0.3)
    
    elif args.mode == 'attack':
        # On ignore les warnings Sklearn car on passe des numpy arrays (attaques) à des modèles entrainés sur DataFrames
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        print("\n[START] TEST DE RÉSILIENCE COMPARATIF (Transferability FGSM)...")
        
        if not HAS_ADVERSARIAL:
            print("ERREUR: Le module src/adversarial.py n'a pas pu être chargé.")
            sys.exit(1)
            
        print(" -> Chargement des modèles...")
        models_to_test = {}
        
        # 1. Deep Learning (Source de l'attaque)
        if os.path.exists(dl_path):
            dl_loaded = DLDetector(input_shape=None) 
            dl_loaded.load_model(dl_path)
            models_to_test["Deep Learning"] = dl_loaded
        else:
            print("ERREUR: Modèle DL requis pour générer les attaques.")
            sys.exit(1)

        # 2. Random Forest
        if os.path.exists(rf_path):
             rf_model.load_model(rf_path)
             models_to_test["Random Forest"] = rf_model

        # 3. XGBoost
        if xgb_model and os.path.exists(xgb_path):
             xgb_model.load_model(xgb_path)
             models_to_test["XGBoost"] = xgb_model

        # 4. Hybrid
        if os.path.exists(rb_path):
             rb_model.load_model(rb_path)
             hybrid_model = HybridDetector(dl_loaded, rb_model)
             models_to_test["Hybride"] = hybrid_model
             models_to_test["Traditionnel"] = rb_model
        
        # 5. AutoEncoder
        # La sauvegarde de l'AE crée un fichier _model.keras et _meta.joblib
        ae_path_model = ae_path.replace(".keras", "_model.keras")
        if os.path.exists(ae_path_model) or os.path.exists(ae_path):
             ae_loaded = AnomalyDetector(input_shape=None)
             ae_loaded.load_model(ae_path)
             models_to_test["AutoEncoder"] = ae_loaded

        # Sampling
        print(" -> Sélection d'un échantillon d'attaques (max 1000)...")
        import numpy as np
        
        sample_size = 1000
        if len(X_val) > sample_size:
            indices = np.random.choice(len(X_val), sample_size, replace=False)
            X_sample = X_val.iloc[indices]
            y_sample = y_val.iloc[indices]
        else:
            X_sample = X_val
            y_sample = y_val

        # Lancement du test comparatif
        results = test_robustness(
            dl_loaded.model, # Source (White-box)
            models_to_test,  # Cibles (Transferability)
            X_sample.values if hasattr(X_sample, 'values') else X_sample, 
            y_sample.values if hasattr(y_sample, 'values') else y_sample, 
            epsilon=0.1
        )
        
        print("\n" + "="*80)
        print(f" RÉSULTATS DU TEST DE RÉSILIENCE (Transferability FGSM epsilon=0.1)")
        print("="*80)
        print(f"{'MODÈLE':<25} | {'TAUX DE SUCCÈS (Evasion)':<15} | {'RÉSILIENCE'}")
        print("-" * 80)
        
        for name, res in results.items():
            if res["Total"] == 0: rate = 0.0
            else: rate = res["Success"] / res["Total"]
            
            resilience_score = 1.0 - rate
            if resilience_score > 0.9: grade = "EXCELLENTE"
            elif resilience_score > 0.7: grade = "BONNE"
            elif resilience_score > 0.5: grade = "MOYENNE"
            else: grade = "CRITIQUE"
            
            print(f"{name:<25} | {rate:.2%}           | {grade}")
            
        print("="*80)
        print("Note: 'Taux de Succès' = % d'attaques qui ont réussi à passer inaperçues.")
        print("Les attaques ont été générées sur le Deep Learning (White-Box) et testées sur les autres.")