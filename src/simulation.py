import time
import random
import sys
import os
import numpy as np
import pandas as pd
import contextlib

# --- COULEURS ET STYLES (ANSI) ---
RED = '\033[91m'     # Pour les attaques / Block / Danger
GREEN = '\033[92m'   # Pour le trafic Safe / Allow / Succès
YELLOW = '\033[93m'  # Pour les avertissements / Latence moyenne
BLUE = '\033[94m'    # Pour les infos neutres
CYAN = '\033[96m'    # Pour la déco
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'
BG_RED = '\033[41m'  # Fond rouge pour alerte critique

# --- UTILITAIRE POUR MASQUER LES LOGS TENSORFLOW ---
@contextlib.contextmanager
def suppress_output():
    """Masque stdout et stderr pour empêcher Keras de casser l'affichage."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class TrafficSimulator:
    def __init__(self, models, X_test, y_test, labels_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.labels_test = labels_test 

    def _generate_fake_metadata(self, attack_name):
        src_ip = f"192.168.{random.randint(10, 50)}.{random.randint(2, 254)}"
        if isinstance(attack_name, str):
            if "SSH" in attack_name: port = 22
            elif "HTTP" in attack_name or "DoS" in attack_name: port = 80
            elif "FTP" in attack_name: port = 21
            else: port = random.randint(1024, 65535)
        else:
            port = random.randint(1024, 65535)
        return src_ip, port

    def run(self, num_packets=20, delay=0.8):
        # --- EN-TÊTE ---
        print("\n" * 2)
        print(f"{CYAN}" + "="*155 + f"{RESET}")
        print(f"{BOLD}{WHITE}   🛡️  LIVE THREAT MONITORING - ANALYSE MULTI-MODÈLES IA   {RESET}")
        print(f"{CYAN}" + "="*155 + f"{RESET}")
        
        header = (
            f"{BOLD}TIME      | "
            f"{'SOURCE IP':<15} | "
            f"{'PORT':<5} | "
            f"{'NATURE DU PAQUET (VÉRITÉ)':<32} | "
            f"{'RANDOM FOREST':<22} | "
            f"{'DEEP LEARNING':<22} | "
            f"{'HYBRIDE':<22}{RESET}"
        )
        print(header)
        print(f"{CYAN}" + "-"*155 + f"{RESET}")

        indices = list(range(len(self.X_test)))
        random.shuffle(indices)
        selected_indices = indices[:num_packets]

        # Initialisation des statistiques détaillées
        stats = {
            name: {
                "correct": 0, 
                "total": 0, 
                "latencies": [],
                "false_negatives": 0, # Attaques ratées (DANGER)
                "false_positives": 0  # Fausses alertes (Bruit)
            } 
            for name in self.models.keys()
        }

        for i in selected_indices:
            row = self.X_test.iloc[[i]]
            is_attack = self.y_test.iloc[i] == 1
            real_label_name = self.labels_test.iloc[i]
            
            timestamp = time.strftime("%H:%M:%S")
            src_ip, port = self._generate_fake_metadata(real_label_name)
            
            if is_attack:
                label_display = f"☣️  ATTAQUE ({str(real_label_name)[:15]})"
                packet_info = f"{RED}{label_display:<32}{RESET}"
            else:
                packet_info = f"{GREEN}{'✓ SAFE TRAFFIC':<32}{RESET}"

            line_start = f"{timestamp} | {src_ip:<15} | {str(port):<5} | {packet_info} | "
            print(line_start, end="", flush=True)

            results_str = []
            
            for model_name, model in self.models.items():
                t0 = time.perf_counter()
                with suppress_output():
                    pred = model.predict(row)
                lat = (time.perf_counter() - t0) * 1000 
                
                # --- CORRECTION DE TYPE (Liste/Array -> Scalaire) ---
                if isinstance(pred, list):
                    pred = pred[0]
                elif isinstance(pred, np.ndarray): 
                    pred = pred.item()
                
                # Conversion explicite en entier pour comparaison sûre
                pred = int(pred)
                truth = int(self.y_test.iloc[i])
                
                # --- CALCUL DES STATISTIQUES ---
                # 1. Latence
                stats[model_name]["latencies"].append(lat)
                stats[model_name]["total"] += 1
                
                # 2. Précision
                correct = (pred == truth)
                if correct: 
                    stats[model_name]["correct"] += 1
                
                # 3. Analyse des erreurs (Confusion Matrix live)
                # Faux Négatif : C'était une attaque (is_attack=True) mais prédit Normal (pred=0)
                if truth == 1 and pred == 0:
                    stats[model_name]["false_negatives"] += 1
                # Faux Positif : C'était Safe mais prédit Attaque
                elif truth == 0 and pred == 1:
                    stats[model_name]["false_positives"] += 1

                # --- AFFICHAGE ---
                if pred == 1: 
                    action = f"{BG_RED}{WHITE} BLOCK {RESET}"
                    pad_width = 36 
                else:         
                    action = f"{GREEN} ALLOW {RESET}"
                    pad_width = 31
                
                status_icon = "✅" if correct else "❌"
                cell = f"{action} {lat:5.1f}ms {status_icon}"
                results_str.append(f"{cell:<{pad_width}}")

            print("| ".join(results_str))
            time.sleep(delay)

        # --- RAPPORT DE PERFORMANCE DÉTAILLÉ ---
        print(f"{CYAN}" + "="*155 + f"{RESET}")
        print(f"\n{BOLD}{WHITE}📊 RAPPORT DE PERFORMANCE FINAL (Sur {len(selected_indices)} paquets){RESET}")
        print(f"{'MODÈLE':<20} | {'PRÉCISION':<12} | {'LATENCE MOY.':<15} | {'ATTAQUES RATÉES (DANGER)':<25} | {'FAUSSES ALERTES'}")
        print("-" * 110)

        for model, data in stats.items():
            if data["total"] > 0:
                # Calculs
                acc = (data["correct"] / data["total"]) * 100
                avg_lat = sum(data["latencies"]) / len(data["latencies"])
                missed_attacks = data["false_negatives"]
                false_alarms = data["false_positives"]

                # Couleurs Précision
                if acc > 95: col_acc = GREEN
                elif acc > 80: col_acc = YELLOW
                else: col_acc = RED

                # Couleurs Latence
                if avg_lat < 1.0: col_lat = GREEN      # Très rapide
                elif avg_lat < 50.0: col_lat = YELLOW  # Acceptable
                else: col_lat = RED                    # Lent

                # Couleur Danger (Attaques ratées)
                if missed_attacks == 0: col_danger = GREEN
                else: col_danger = RED + BOLD

                print(f"{model:<20} | {col_acc}{acc:6.2f}%{RESET}     | {col_lat}{avg_lat:8.2f} ms{RESET}    | {col_danger}{missed_attacks:^25}{RESET} | {false_alarms}")

        print("\n")