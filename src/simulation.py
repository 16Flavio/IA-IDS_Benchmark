import time
import random
import sys
import os
import numpy as np
import pandas as pd
import contextlib

# --- COULEURS ET STYLES (ANSI) ---
RED = '\033[91m'     # Pour les attaques / Block
GREEN = '\033[92m'   # Pour le trafic Safe / Allow
YELLOW = '\033[93m'  # Pour les avertissements
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
        self.labels_test = labels_test # Les noms réels (ex: "DoS Hulk")

    def _generate_fake_metadata(self, attack_name):
        """Génère des fausses métadonnées (IP, Port) cohérentes pour le réalisme."""
        src_ip = f"192.168.{random.randint(10, 50)}.{random.randint(2, 254)}"
        
        # Ports cohérents avec le type d'attaque (simulation)
        if isinstance(attack_name, str):
            if "SSH" in attack_name: port = 22
            elif "HTTP" in attack_name or "DoS" in attack_name: port = 80
            elif "FTP" in attack_name: port = 21
            else: port = random.randint(1024, 65535)
        else:
            port = random.randint(1024, 65535)
        
        return src_ip, port

    def run(self, num_packets=20, delay=0.8):
        # --- EN-TÊTE DU TABLEAU DE BORD ---
        print("\n" * 2)
        print(f"{CYAN}" + "="*155 + f"{RESET}")
        print(f"{BOLD}{WHITE}   🛡️  LIVE THREAT MONITORING - ANALYSE MULTI-MODÈLES IA   {RESET}")
        print(f"{CYAN}" + "="*155 + f"{RESET}")
        
        # En-têtes des colonnes alignés
        # On définit des largeurs fixes pour éviter le décalage
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

        stats = {name: {"correct": 0, "total": 0} for name in self.models.keys()}

        for i in selected_indices:
            row = self.X_test.iloc[[i]]
            is_attack = self.y_test.iloc[i] == 1
            real_label_name = self.labels_test.iloc[i]
            
            # 1. Génération des métadonnées visuelles
            timestamp = time.strftime("%H:%M:%S")
            src_ip, port = self._generate_fake_metadata(real_label_name)
            
            # Affichage de la colonne "VÉRITÉ"
            if is_attack:
                # Si c'est une attaque, on l'affiche en rouge vif avec son nom tronqué si trop long
                label_display = f"☣️  ATTAQUE ({str(real_label_name)[:15]})"
                packet_info = f"{RED}{label_display:<32}{RESET}"
            else:
                # Si c'est safe, en vert
                packet_info = f"{GREEN}{'✓ SAFE TRAFFIC':<32}{RESET}"

            # Début de la ligne (Infos Paquet)
            # flush=True force l'affichage immédiat avant les calculs
            line_start = f"{timestamp} | {src_ip:<15} | {str(port):<5} | {packet_info} | "
            print(line_start, end="", flush=True)

            # 2. Analyse par chaque modèle
            results_str = []
            
            for model_name, model in self.models.items():
                # Mesure latence
                t0 = time.perf_counter()
                
                # CRITIQUE : On masque la sortie standard pour que Keras ne pollue pas l'affichage
                with suppress_output():
                    pred = model.predict(row)
                
                lat = (time.perf_counter() - t0) * 1000 # ms
                
                if isinstance(pred, np.ndarray): pred = pred.item()
                
                # Vérification
                correct = (pred == self.y_test.iloc[i])
                if correct: stats[model_name]["correct"] += 1
                stats[model_name]["total"] += 1

                # Mise en forme de la décision : ALLOW (Vert) ou BLOCK (Rouge)
                if pred == 1: # Le modèle dit "Attaque" -> BLOCK
                    action = f"{BG_RED}{WHITE} BLOCK {RESET}"
                    # CORRECTION ALIGNEMENT :
                    # BLOCK (Fond Rouge) a 14 caractères invisibles ANSI.
                    # On veut une largeur visible finale de 22.
                    # Donc largeur totale string = 22 (visible) + 14 (invisible) = 36.
                    pad_width = 36 
                else:         # Le modèle dit "Normal" -> ALLOW
                    action = f"{GREEN} ALLOW {RESET}"
                    # ALLOW (Vert) a 9 caractères invisibles ANSI.
                    # Largeur totale string = 22 (visible) + 9 (invisible) = 31.
                    pad_width = 31
                
                # Icône de succès (Le modèle a-t-il raison ?)
                status_icon = "✅" if correct else "❌"
                
                # Construction de la cellule du modèle
                # On force la latence sur 5.1f pour éviter le décalage si > 10ms ou > 100ms
                cell = f"{action} {lat:5.1f}ms {status_icon}"
                results_str.append(f"{cell:<{pad_width}}")

            # Affichage des décisions des modèles
            print("| ".join(results_str))
            
            # Petit délai pour l'effet "Temps Réel"
            time.sleep(delay)

        # --- RÉSUMÉ DE LA SESSION ---
        print(f"{CYAN}" + "="*155 + f"{RESET}")
        print(f"\n{BOLD}📊 RAPPORT DE PERFORMANCE DE LA SESSION :{RESET}")
        for model, data in stats.items():
            if data["total"] > 0:
                acc = (data["correct"] / data["total"]) * 100
                if acc > 95: color = GREEN
                elif acc > 80: color = YELLOW
                else: color = RED
                print(f"   • {model:<20} : {color}{acc:.1f}%{RESET} de décisions correctes sur ce flux.")
        print("\n")