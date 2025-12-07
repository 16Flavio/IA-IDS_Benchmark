import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import pandas as pd
import time
import sys
from src.data_loader import DataLoader
from src.methods import RuleBasedDetector, MLDetector, DLDetector, HybridDetector
from src.evaluation import Evaluator
from src.simulation import TrafficSimulator
from sklearn.metrics import accuracy_score, f1_score
import datetime

def update_readme(results_df):
    """
    Met à jour le fichier README.md à la racine du projet avec les nouveaux résultats.
    Cela permet d'avoir une documentation toujours à jour sur GitHub.
    """
    # Construction du tableau Markdown
    md_table = "| Modèle | Précision | F1-Score | Temps (s) |\n"
    md_table += "| :--- | :--- | :--- | :--- |\n"
    
    for _, row in results_df.iterrows():
        t = row['Temps (s)']
        # Formatage propre du temps (évite d'afficher "0.123456789")
        temps_str = f"{t:.4f}" if isinstance(t, float) else str(t)
        md_table += f"| **{row['Modèle']}** | {row['Précision']:.2%} | {row['F1-Score']:.4f} | {temps_str} |\n"

    # Le contenu statique du README
    readme_content = f"""# 🛡️ IDS-Benchmark: Comparatif ML/DL pour la Cybersécurité

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

## 📈 Résultats (Dernière mise à jour : {datetime.datetime.now().strftime("%d/%m/%Y à %H:%M")})
{md_table}

## 🔍 Analyse
* Consultez le dossier `/results` pour visualiser les **Matrices de Confusion** et les **Courbes ROC**.
* Consultez `notebooks/01_Data_Exploration.ipynb` pour l'analyse exploratoire des données.
"""
    # Écriture dans le fichier
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("\n[AUTO] README.md mis à jour avec succès !")

if __name__ == "__main__":
    # Configuration des arguments ligne de commande
    parser = argparse.ArgumentParser(description="AI IDS Benchmark")
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'sim'], default='train', help="Mode: 'train', 'eval' ou sim")
    parser.add_argument('--dataset', type=str, default='cic_ids2017', help="Dataset: 'nsl_kdd' ou 'cic_ids2017'")
    args = parser.parse_args()

    # Dossier pour sauvegarder les modèles
    MODEL_DIR = "models"
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print(f"--- BENCHMARK CYBERSÉCURITÉ : {args.dataset.upper()} | MODE : {args.mode.upper()} ---")

    # Chargement des données
    loader = DataLoader(args.dataset)
    X_train, X_test, y_train, y_test, labels_test = loader.load_data()

    # Initialisation des modèles
    # Note: Pour le DL, on donne la shape uniquement si on va créer un nouveau modèle
    rf_model = MLDetector()
    dl_model = DLDetector(input_shape=X_train.shape[1]) 
    rb_model = RuleBasedDetector()

    # Chemins de sauvegarde
    rf_path = os.path.join(MODEL_DIR, f"rf_{args.dataset}.joblib")
    dl_path = os.path.join(MODEL_DIR, f"dl_{args.dataset}.keras")

    # --- MODE ENTRAINEMENT ---
    if args.mode == 'train':
        print("\n[START] Entraînement des modèles...")
        
        # Random Forest
        print(" -> Random Forest...")
        rf_model.train(X_train, y_train)
        rf_model.save_model(rf_path)
        
        # Deep Learning
        print(" -> Deep Learning...")
        dl_model.train(X_train, y_train)
        dl_model.save_model(dl_path)
        
        print("\nEntraînement terminé. Modèles sauvegardés dans /models")

    # --- MODE EVALUATION ---
    elif args.mode == 'eval':
        print("\n[START] Évaluation complète...")
        evaluator = Evaluator(output_dir='results')
        
        # Chargement des modèles
        print(" -> Chargement des modèles...")
        rf_model.load_model(rf_path)
        
        # Pour le DL, on recharge le fichier .keras qui contient toute l'architecture
        # On recrée une instance vide et on charge le fichier
        dl_loaded = DLDetector(input_shape=None) 
        dl_loaded.load_model(dl_path)
        
        # Modèle Hybride (basé sur le DL chargé)
        hybrid_model = HybridDetector(dl_loaded)
        
        models_to_eval = {
            "Random Forest": rf_model,
            "Deep Learning": dl_loaded,
            "Hybride": hybrid_model,
            "Traditionnel": rb_model
        }
        
        results = []

        for name, model in models_to_eval.items():
            print(f" -> Testing {name}...")
            start = time.time()
            
            preds = model.predict(X_test)
            if name == "Deep Learning": preds = preds.flatten()
            
            duration = time.time() - start
            
            # Métriques de base
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            
            results.append({"Modèle": name, "Précision": acc, "F1-Score": f1, "Temps (s)": duration})
            
            # Graphiques avancés via evaluation.py
            evaluator.plot_confusion_matrix(y_test, preds, name)
            evaluator.save_report(y_test, preds, name)
            
            # Courbes ROC (si le modèle supporte predict_proba)
            try:
                probs = model.predict_proba(X_test)
                # Gestion format probas (RF renvoie [N, 2], DL renvoie [N, 1])
                if name == "Deep Learning":
                    probs = probs.flatten()
                elif hasattr(probs, "shape") and probs.shape[1] == 2:
                    probs = probs[:, 1]
                
                evaluator.plot_roc_curve(y_test, probs, name)
            except Exception as e:
                print(f"   (Pas de ROC pour {name}: {e})")

        # Affichage final console
        res_df = pd.DataFrame(results)
        print("\n--- RÉSULTATS FINAUX ---")
        print(res_df.sort_values(by="F1-Score", ascending=False))
        update_readme(res_df)
        print(f"\nGraphiques sauvegardés dans le dossier /results")
    
    elif args.mode == 'sim':
        print("\n[START] Simulateur Temps Réel...")
        
        if not os.path.exists(rf_path):
            print(f"ERREUR: Modèles non trouvés.")
            sys.exit(1)

        rf_model.load_model(rf_path)
        dl_loaded = DLDetector(input_shape=None) 
        dl_loaded.load_model(dl_path)
        hybrid_model = HybridDetector(dl_loaded)
        
        simulation_models = {
            "RandomForest": rf_model,
            "DeepLearn": dl_loaded,
            "Hybride": hybrid_model
        }
        
        # On passe 'labels_test' au simulateur pour afficher les noms d'attaques
        sim = TrafficSimulator(simulation_models, X_test, y_test, labels_test)
        sim.run(num_packets=30, delay=1.5)