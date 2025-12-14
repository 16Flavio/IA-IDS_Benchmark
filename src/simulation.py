import time
import random
import sys
import os
import numpy as np
import pandas as pd
import contextlib
import re
from datetime import datetime

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

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
    def __init__(self, models_dict, X_val, y_val, labels_val):
        self.models = models_dict # Dict de modèles {"RandomForest": model, ...}
        self.X_val = X_val
        self.y_val = y_val.reset_index(drop=True)
        self.labels_val = labels_val.reset_index(drop=True)
        
        # ACTIVE DEFENSE: Cache des attaquants pour les ré-utiliser (Simulation d'attaque persistante)
        self.known_attackers = [] 
        
        # Dictionnaire pour traduire les features techniques en langage humain et pédagogique
        self.feature_map = {
            'Flow Duration': 'Durée de Connexion Anormale (Possible Scan Lent)',
            'Total Fwd Packets': 'Volume d\'Upload Suspect (Exfiltration ?)',
            'Total Backward Packets': 'Volume de Download Suspect',
            'Total Length of Fwd Packets': 'Taille Upload Inhabituelle',
            'Fwd Packet Length Max': 'Paquet Trop Volumineux',
            'Destination Port': 'Port Cible Suspect (Scan/Exploit)',
            'dst_host_count': 'Trafic vers Hôte Saturé',
            'src_bytes': 'Volume Source Anormal',
            'dst_bytes': 'Volume Destination Anormal',
            'count': 'Fréquence de Connexion Élevée (Brute Force ?)',
            'srv_count': 'Services Multiples (Scan ?)',
        }

    def _generate_fake_metadata(self, is_attack):
        """Génère des fausses métadonnées (IP, Port) pour le réalisme."""
        port = np.random.randint(1024, 65535)
        
        if is_attack:
            # ACTIVE DEFENSE: 50% de chance de réutiliser une IP d'attaquant connu
            if self.known_attackers and np.random.rand() > 0.5:
                ip = np.random.choice(self.known_attackers)
            else:
                ip = f"192.168.1.{np.random.randint(100, 200)}"
                self.known_attackers.append(ip) # On le mémorise
        else:
            ip = f"10.0.0.{np.random.randint(2, 254)}"
            
        return ip, port

    def _get_model_proba(self, model, row):
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(row)
        elif hasattr(model, 'dl_model'): 
            return model.dl_model.predict_proba(row)
        else:
            return None

    def _get_true_explanation(self, model, row, model_name):
        """Moteur XAI : Analyse causale détaillée."""
        if "AutoEncoder" in model_name:
             # Cas Spécial Zero-Day : On ne regarde pas la proba mais la MSE
             try:
                 mse = model.predict_proba(row)[0] # Retourne la MSE
                 thresh = getattr(model, "threshold", 0.05)
                 if mse > thresh:
                     return f"Comportement Inconnu (Zero-Day)"
                 else:
                     return ""
             except Exception as e:
                 print(f"DEBUG ERROR XAI: {e}")
                 return "Erreur XAI AE"

        # Cas Classique (RF, DL, Hybride)
        with suppress_output():
            base_pred = self._get_model_proba(model, row)
        
        if base_pred is None: return ""

        # Gestion des différentes formes de sortie (Proba 1D, 2D, Scalar)
        if isinstance(base_pred, list): base_prob = base_pred[0]
        elif hasattr(base_pred, "shape") and base_pred.shape == (1, 2): base_prob = base_pred[0][1]
        elif hasattr(base_pred, "shape") and base_pred.shape == (1, 1): base_prob = base_pred[0][0]
        elif hasattr(base_pred, "item"): base_prob = base_pred.item()
        else: base_prob = float(base_pred)

        if base_prob < 0.5: return "" # Pas d'explication si considéré Safe

        # 1. Analyse Hybride (Logique Règle)
        if "Hybride" in model_name:
            is_cic = 'Destination Port' in row.columns
            # Si dans la zone d'incertitude (entre 25% et 75%), c'est probablement une règle qui a tranché
            if 0.20 <= base_prob <= 0.80:
                if is_cic:
                    if 'Flow Duration' in row and row['Flow Duration'].item() > 0.5: return "Règle: Durée Trop Longue"
                    if 'Total Fwd Packets' in row and row['Total Fwd Packets'].item() > 0.5: return "Règle: Volume Excessif"
                return "Règle: Seuil Statistique Dépassé"

        # 2. Analyse Perturbation (Deep Learning & RF)
        # On cherche la feature qui fait le plus baisser la probabilité d'attaque si on la met à 0
        significant_features = row.columns[row.abs().gt(0.1).any()].tolist() # Optimisation: on ne teste que les features > 0.1
        impacts = {}
        
        # Limite pour performance
        if len(significant_features) > 10: 
            significant_features = significant_features[:10]

        for feature in significant_features:
            perturbed_row = row.copy()
            perturbed_row[feature] = 0.0 # On neutralise la feature (valeur moyenne car centré réduit)
            
            with suppress_output():
                new_pred = self._get_model_proba(model, perturbed_row)
            
            if new_pred is None: continue

            if isinstance(new_pred, list): new_prob = new_pred[0]
            elif hasattr(new_pred, "shape") and new_pred.shape == (1, 2): new_prob = new_pred[0][1]
            elif hasattr(new_pred, "shape") and new_pred.shape == (1, 1): new_prob = new_pred[0][0]
            elif hasattr(new_pred, "item"): new_prob = new_pred.item()
            else: new_prob = float(new_pred)
            
            impacts[feature] = base_prob - new_prob # Chute de confiance

        if not impacts: return "Motif Global"
            
        best_feature = max(impacts, key=impacts.get)
        confidence_drop = impacts[best_feature]
        
        if confidence_drop < 0.01: return "Pattern Complexe"
        
        # --- ENRICHISSEMENT DE L'EXPLICATION ---
        human_name = self.feature_map.get(best_feature, best_feature[:15])
        return f"{human_name}"

    def run(self, num_packets=50, delay=0.5):
        print(f"\n{Colors.HEADER}--- DÉMARRAGE DU TRAFFIC SIMULATOR (DASHBOARD SOC) ---{Colors.ENDC}")
        print(f"{Colors.BOLD}Simulation de {num_packets} paquets en temps réel...{Colors.ENDC}\n")
        # Header élargi pour XAI
        print(f"{'TIMESTAMP':<10} | {'SOURCE IP':<15} | {'TYPE':<12} | {'DÉTECTION (MODELE)':<40} | {'ACTION':<10} | {'RAISON (XAI)':<20}")
        print("-" * 125)

        stats = {name: {"Blocked": 0, "Allowed": 0, "Firewall": 0} for name in self.models.keys()}
        
        if len(self.X_val) == 0:
            print("Erreur: Dataset vide pour la simulation.")
            return

        indices = np.random.choice(len(self.X_val), min(num_packets, len(self.X_val)), replace=False)

        for i in indices:
            row = self.X_val.iloc[[i]] # DataFrame
            actual_label = self.y_val[i]
            attack_name = self.labels_val[i] if actual_label == 1 else "Normal"
            is_attack = (actual_label == 1)
            
            src_ip, src_port = self._generate_fake_metadata(is_attack)
            timestamp = datetime.now().strftime("%H:%M:%S")

            # --- ACTIVE DEFENSE LOGIC ---
            firewall_model = self.models.get("Traditionnel (Règles)") or self.models.get("Hybride")
            is_firewalled = False
            
            if firewall_model:
                if hasattr(firewall_model, "rb_model") and firewall_model.rb_model: # Cas Hybride
                     if firewall_model.rb_model.is_blocked(src_ip):
                         is_firewalled = True
                elif hasattr(firewall_model, "is_blocked"): # Cas Traditionnel
                     if firewall_model.is_blocked(src_ip):
                         is_firewalled = True

            
            # Affichage pour chaque modèle
            results_display = []
            final_action = f"{Colors.OKGREEN}ALLOW{Colors.ENDC}"
            explanation = "" # Explication globale

            for model_name, model in self.models.items():
                if is_firewalled:
                    # BLOCKED BY FIREWALL
                    pred = 1 
                    if model_name in ["Traditionnel (Règles)", "Hybride"]:
                        stats[model_name]["Firewall"] += 1
                        stats[model_name]["Blocked"] += 1
                        final_action = f"{Colors.FAIL}BLOCK [FW]{Colors.ENDC}"
                        explanation = f"{Colors.WARNING}IP Blacklistée{Colors.ENDC}"
                    else:
                        # Les autres modèles n'ont pas forcément accès au FW
                        # Pour l'affichage, on montre qu'ils sont 'out' ou on simule leur décision
                        # On va dire qu'ils sont bypassés par le FW
                        stats[model_name]["Firewall"] += 1
                        stats[model_name]["Blocked"] += 1
                        
                else:
                    # BLOCKED BY AI
                    try:
                        pred = model.predict(row)
                        # Conversion robuste scalaire
                        if hasattr(pred, "values"): pred = pred.values
                        if hasattr(pred, "flatten"): pred = pred.flatten()[0]
                        elif isinstance(pred, list): pred = pred[0]
                    except:
                        pred = 0

                    if pred == 1:
                        action_str = f"{Colors.FAIL}BLOCK [AI]{Colors.ENDC}"
                        stats[model_name]["Blocked"] += 1
                        final_action = action_str
                        
                        # On calcule l'explication XAI pour ce modèle
                        # On prend la première explication pertinente trouvée pour l'affichage global
                        if not explanation:
                            reason = self._get_true_explanation(model, row, model_name)
                            if reason:
                                explanation = reason

                        # ACTIVE DEFENSE UPDATE
                        if firewall_model and is_attack:
                            if hasattr(firewall_model, "rb_model") and firewall_model.rb_model:
                                firewall_model.rb_model.update_blocklist(src_ip)
                            elif hasattr(firewall_model, "update_blocklist"):
                                firewall_model.update_blocklist(src_ip)

                    else:
                        stats[model_name]["Allowed"] += 1

                # Coloration du nom du modèle
                is_correct = (int(pred) == int(actual_label))
                if is_correct:
                    res_str = f"{Colors.OKGREEN}{model_name}{Colors.ENDC}"
                else:
                    res_str = f"{Colors.FAIL}{model_name}{Colors.ENDC}"
                
                results_display.append(res_str)

            # Formatage de l'affichage ligne
            type_color = Colors.FAIL if is_attack else Colors.OKGREEN
            
            # Formattage de l'explication
            if explanation:
                expl_str = f"{Colors.YELLOW}{explanation}{Colors.ENDC}"
            else:
                expl_str = "-"

            print(f"{timestamp} | {src_ip:<15} | {type_color}{attack_name[:12]:<12}{Colors.ENDC} | {' | '.join(results_display):<60} | {final_action:<10} | {expl_str}")
            
            time.sleep(delay)

        print("-" * 125)
        print(f"{Colors.BOLD}Simulation Terminée.{Colors.ENDC}")
        print("Résumé des Actions :")
        for m, s in stats.items():
            print(f" - {m:<20}: {s['Blocked']} Bloqués (dont {s['Firewall']} par Firewall), {s['Allowed']} Autorisés")
        print("\n")