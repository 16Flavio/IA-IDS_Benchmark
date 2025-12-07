import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataLoader:
    def __init__(self, dataset_name='nsl_kdd'):
        self.dataset_name = dataset_name
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    def load_data(self):
        print(f"Chargement des données locales pour : {self.dataset_name}...")
        if self.dataset_name == 'nsl_kdd':
            X, y, labels = self._load_nsl_kdd()
        elif self.dataset_name == 'cic_ids2017':
            X, y, labels = self._load_cic_ids()
        else:
            raise ValueError("Dataset inconnu. Choisir 'nsl_kdd' ou 'cic_ids2017'.")
        
        X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(X, y, labels, test_size=0.4, random_state=42)
        
        scaler = StandardScaler()
        
        cols = X.columns
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=cols)
        
        return X_train, X_test, y_train, y_test, labels_test

    def _load_nsl_kdd(self):
        # Chemin vers le fichier local
        file_path = os.path.join(self.base_dir, 'data', 'nsl_kdd', 'KDDTrain+.txt')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier introuvable : {file_path}. As-tu téléchargé KDDTrain+.txt ?")

        cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 
                'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
                'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 
                'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
                'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
                'dst_host_srv_rerror_rate', 'attack', 'level']
        
        df = pd.read_csv(file_path, names=cols)
        
        labels = df['attack']

        # Encodage binaire (Normal = 0, Attaque = 1)
        df['label'] = df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # Encodage des strings en nombres
        for col in ['protocol_type', 'service', 'flag']:
            df[col] = LabelEncoder().fit_transform(df[col])
            
        X = df.drop(['attack', 'level', 'label'], axis=1)
        y = df['label']
        
        return X, y, labels

    def _load_cic_ids(self):
        # Chemin vers le fichier local (Wednesday = Attaques DoS / DDoS / Heartbleed)
        file_path = os.path.join(self.base_dir, 'data', 'cic_ids2017', 'Wednesday-workingHours.pcap_ISCX.csv')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier introuvable : {file_path}. Vérifie le dossier data/cic_ids2017/")

        # Le fichier réel est gros, on lit tout (ou tu peux mettre nrows=50000 pour tester vite)
        df = pd.read_csv(file_path)
        
        # NETTOYAGE CRITIQUE POUR CIC-IDS2017
        # Retirer les espaces dans les noms de colonnes (" Label" -> "Label")
        df.columns = df.columns.str.strip()
        
        # Gérer les infinis et les valeurs vides (sinon le Random Forest plante)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        labels = df['Label']

        # Encodage Cible : BENIGN = 0, Le reste (DoS, etc.) = 1
        df['label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        # Sélectionner uniquement les colonnes numériques (pour simplifier le ML)
        # On exclut 'Label' (string) et notre nouvelle target 'label'
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in numeric_cols: numeric_cols.remove('label')
        
        X = df[numeric_cols]
        y = df['label']
        
        # On réduit un peu la taille si besoin pour que ça tourne sur un PC normal
        # X = X.iloc[:100000] 
        # y = y.iloc[:100000]
        
        return X, y, labels