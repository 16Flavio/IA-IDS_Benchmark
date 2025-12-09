import time
import random
import sys
import os
import numpy as np
import pandas as pd
import contextlib
import re

# --- COULEURS---
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'
BG_RED = '\033[41m'

# --- UTILITAIRES D'AFFICHAGE ---
def visible_len(s):
    """Calcule la longueur visible d'une chaîne en ignorant les codes couleurs ANSI."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', s))

def pad_ansi(s, width):
    """Ajoute des espaces à droite pour atteindre la largeur visible demandée."""
    v_len = visible_len(s)
    padding = max(0, width - v_len)
    return s + " " * padding

@contextlib.contextmanager
def suppress_output():
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
        
        # Dictionnaire pour traduire les features techniques en langage humain
        self.feature_map = {
            'Flow Duration': 'Durée Connexion',
            'Total Fwd Packets': 'Volume Upload',
            'Total Backward Packets': 'Volume Download',
            'Total Length of Fwd Packets': 'Taille Upload',
            'Fwd Packet Length Max': 'Paquet Max',
            'Destination Port': 'Port Dest.',
            'dst_host_count': 'Fréq. Hôte',
            'src_bytes': 'Octets Source',
            'dst_bytes': 'Octets Dest.',
            'count': 'Fréq. Connexion'
        }

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

    def _get_model_proba(self, model, row):
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(row)
        elif hasattr(model, 'dl_model'): 
            return model.dl_model.predict_proba(row)
        else:
            return None

    def _get_true_explanation(self, model, row, model_name):
        """Moteur XAI : Analyse causale détaillée."""
        with suppress_output():
            base_pred = self._get_model_proba(model, row)
        
        if base_pred is None: return ""

        if isinstance(base_pred, list): base_prob = base_pred[0]
        elif base_pred.shape[1] == 2: base_prob = base_pred[0][1]
        else: base_prob = base_pred[0][0]

        if base_prob < 0.5: return "" # Pas d'explication si Safe

        # 1. Analyse Hybride (Logique Règle)
        if "Hybride" in model_name:
            is_cic = 'Destination Port' in row.columns
            # Si dans la zone d'incertitude (entre 25% et 75%), c'est probablement une règle
            if 0.25 <= base_prob <= 0.75:
                if is_cic:
                    if row['Flow Duration'].item() > 0.5: return "Règle: Durée > Seuil"
                    if row['Total Fwd Packets'].item() > 1.0: return "Règle: Flood Paquets"
                else:
                    if row.get('dst_host_count', 0).item() > 0.5: return "Règle: Fréq. Hôte"
                return "Règle: Sécurité"

        # 2. Analyse Perturbation (Deep Learning & RF)
        significant_features = row.columns[row.abs().gt(0.5).any()].tolist()
        impacts = {}
        
        for feature in significant_features:
            perturbed_row = row.copy()
            perturbed_row[feature] = 0.0 # On neutralise la feature
            
            with suppress_output():
                new_pred = self._get_model_proba(model, perturbed_row)
            
            if new_pred is None: continue

            if isinstance(new_pred, list): new_prob = new_pred[0]
            elif new_pred.shape[1] == 2: new_prob = new_pred[0][1]
            else: new_prob = new_pred[0][0]
            
            impacts[feature] = base_prob - new_prob

        if not impacts: return "Motif Global"
            
        best_feature = max(impacts, key=impacts.get)
        confidence_drop = impacts[best_feature]
        
        if confidence_drop < 0.05: return "Combinaison Complexe"
        
        # --- ENRICHISSEMENT DE L'EXPLICATION ---
        # On regarde la valeur brute (Z-Score) pour dire si c'est "Trop haut" ou "Anormal"
        val = row[best_feature].item()
        human_name = self.feature_map.get(best_feature, best_feature[:10])
        
        if val > 0: qualifier = "Trop Haut"
        else: qualifier = "Anormal"
        
        return f"{human_name} ({qualifier})"

    def run(self, num_packets=20, delay=1.0):
        # En-tête élargi
        print("\n" * 2)
        print(f"{CYAN}" + "="*185 + f"{RESET}")
        print(f"{BOLD}{WHITE}   LIVE THREAT MONITORING - XAI ENABLED (Alignement & Détails)   {RESET}")
        print(f"{CYAN}" + "="*185 + f"{RESET}")
        
        # Largeurs de colonnes fixes (Visibles)
        w_time = 10
        w_nat = 30
        w_rf = 34
        w_dl = 34
        w_hyb = 34

        header = (
            pad_ansi(f"{BOLD}TIME", w_time) + "| " +
            pad_ansi("NATURE (VÉRITÉ)", w_nat) + "| " +
            pad_ansi("RANDOM FOREST", w_rf) + "| " +
            pad_ansi("DEEP LEARNING", w_dl) + "| " +
            pad_ansi("HYBRIDE", w_hyb) + RESET
        )
        print(header)
        print(f"{CYAN}" + "-"*185 + f"{RESET}")

        indices = list(range(len(self.X_test)))
        random.shuffle(indices)
        selected_indices = indices[:num_packets]
        
        # Stats pour le rapport final
        stats = {name: {"correct": 0, "total": 0, "latencies": [], "fn": 0, "fp": 0} for name in self.models.keys()}

        for i in selected_indices:
            row = self.X_test.iloc[[i]]
            is_attack = self.y_test.iloc[i] == 1
            real_label_name = self.labels_test.iloc[i]
            
            timestamp = time.strftime("%H:%M:%S")
            src_ip, port = self._generate_fake_metadata(real_label_name)
            
            # VÉRITÉ TERRAIN
            if is_attack:
                label_txt = f"ATTAQUE ({str(real_label_name)[:15]})"
                packet_info = f"{RED}{label_txt}{RESET}"
            else:
                packet_info = f"{GREEN}SAFE TRAFFIC{RESET}"

            line_start = pad_ansi(f"{timestamp}", w_time) + "| " + pad_ansi(packet_info, w_nat) + "| "
            print(line_start, end="", flush=True)

            results_str = []
            
            for model_name, model in self.models.items():
                t0 = time.perf_counter()
                with suppress_output():
                    pred = model.predict(row)
                lat = (time.perf_counter() - t0) * 1000 
                
                # Conversion & Stats
                if isinstance(pred, list): pred = pred[0]
                elif isinstance(pred, np.ndarray): pred = pred.item()
                pred = int(pred)
                truth = int(self.y_test.iloc[i])
                
                stats[model_name]["latencies"].append(lat)
                stats[model_name]["total"] += 1
                if pred == truth: stats[model_name]["correct"] += 1
                if truth == 1 and pred == 0: stats[model_name]["fn"] += 1
                elif truth == 0 and pred == 1: stats[model_name]["fp"] += 1

                # Construction Cellule
                reason_str = ""
                if pred == 1: # BLOCK
                    action = f"{BG_RED}{WHITE} BLOCK {RESET}"
                    reason = self._get_true_explanation(model, row, model_name)
                    reason_str = f"{YELLOW}[{reason}]{RESET}"
                else:         # ALLOW
                    action = f"{GREEN} ALLOW {RESET}"
                
                status_icon = "✅" if pred == truth else "❌"
                
                # Assemblage avec alignement strict
                # Contenu : Action + Raison + Icone
                content = f"{action} {reason_str}"
                
                # On pad le contenu principal à (largeur - 3 chars pour l'icone)
                # w_model - 2 (espaces) - 2 (icone) = w_model - 4
                padded_content = pad_ansi(content, w_rf - 4)
                full_cell = f"{padded_content} {status_icon}"
                
                results_str.append(full_cell)

            print(" | ".join(results_str))
            time.sleep(0.1) # Un peu plus rapide car le calcul XAI ajoute déjà de la latence

        # --- RAPPORT DE FIN ---
        print(f"{CYAN}" + "="*185 + f"{RESET}")
        print(f"\n{BOLD}{WHITE}RAPPORT DE PERFORMANCE FINAL (Sur {len(selected_indices)} paquets){RESET}")
        
        # En-têtes du rapport
        r_mod = 20
        r_acc = 12
        r_lat = 15
        r_dang = 25
        r_fa = 15
        
        rep_header = (
            pad_ansi("MODÈLE", r_mod) + "| " +
            pad_ansi("PRÉCISION", r_acc) + "| " +
            pad_ansi("LATENCE MOY.", r_lat) + "| " +
            pad_ansi("ATTAQUES RATÉES (Danger)", r_dang) + "| " +
            pad_ansi("FAUSSES ALERTES", r_fa)
        )
        print(rep_header)
        print("-" * 100)

        for model, data in stats.items():
            if data["total"] > 0:
                acc = (data["correct"] / data["total"]) * 100
                avg_lat = sum(data["latencies"]) / len(data["latencies"])
                
                # Couleurs
                c_acc = GREEN if acc > 95 else (YELLOW if acc > 80 else RED)
                c_lat = GREEN if avg_lat < 1.0 else (YELLOW if avg_lat < 50.0 else RED)
                c_dang = GREEN if data["fn"] == 0 else (RED + BOLD)
                
                row_str = (
                    pad_ansi(f"{model}", r_mod) + "| " +
                    pad_ansi(f"{c_acc}{acc:6.2f}%{RESET}", r_acc) + "| " +
                    pad_ansi(f"{c_lat}{avg_lat:6.2f} ms{RESET}", r_lat) + "| " +
                    pad_ansi(f"{c_dang}{str(data['fn']):^25}{RESET}", r_dang) + "| " +
                    pad_ansi(f"{str(data['fp']):^15}", r_fa)
                )
                print(row_str)
        print("\n")