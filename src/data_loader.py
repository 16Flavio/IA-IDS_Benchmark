import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataLoader:
    def __init__(self, dataset_name='nsl_kdd'):
        self.dataset_name = dataset_name
        # On remonte de deux niveaux depuis ce fichier pour trouver la racine
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.processed_dir = os.path.join(self.base_dir, 'data', 'processed', self.dataset_name)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
    def load_data(self):
        print(f"Chargement des données pour : {self.dataset_name}...")
        
        # Check if pre-processed data exists
        split_files = ['X_train.joblib', 'X_test.joblib', 'X_val.joblib', 
                       'y_train.joblib', 'y_test.joblib', 'y_val.joblib', 
                       'labels_val.joblib', 'labels_test.joblib']
        
        all_exist = all(os.path.exists(os.path.join(self.processed_dir, f)) for f in split_files)
        
        if all_exist:
            print("   -> Chargement des données pré-traitées sauvegardées...")
            import joblib
            X_train = joblib.load(os.path.join(self.processed_dir, 'X_train.joblib'))
            X_test = joblib.load(os.path.join(self.processed_dir, 'X_test.joblib'))
            X_val = joblib.load(os.path.join(self.processed_dir, 'X_val.joblib'))
            y_train = joblib.load(os.path.join(self.processed_dir, 'y_train.joblib'))
            y_test = joblib.load(os.path.join(self.processed_dir, 'y_test.joblib'))
            y_val = joblib.load(os.path.join(self.processed_dir, 'y_val.joblib'))
            labels_test = joblib.load(os.path.join(self.processed_dir, 'labels_test.joblib'))
            labels_val = joblib.load(os.path.join(self.processed_dir, 'labels_val.joblib'))
            
            print(f"   -> Données chargées. Train: {X_train.shape}, Test: {X_test.shape}, Val: {X_val.shape}")
            return X_train, X_test, X_val, y_train, y_test, y_val, labels_test, labels_val

        # Sinon, on charge depuis les CSV bruts
        print("   -> Création des splits (60% Train, 20% Test, 20% Val)...")
        if self.dataset_name == 'nsl_kdd':
            X, y, labels = self._load_nsl_kdd()
        elif self.dataset_name == 'cic_ids2017':
            X, y, labels = self._load_cic_ids()
        else:
            raise ValueError("Dataset inconnu.")

        # 1. Split Train (60%) vs Temp (40%)
        X_train, X_temp, y_train, y_temp, _, labels_temp = train_test_split(
            X, y, labels, test_size=0.4, random_state=42, stratify=y
        )
        
        # 2. Split Temp en Test (50% de temp -> 20% total) et Val (50% de temp -> 20% total)
        X_test, X_val, y_test, y_val, labels_test, labels_val = train_test_split(
            X_temp, y_temp, labels_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print("Normalisation des données (StandardScaler)...")
        scaler = StandardScaler()
        cols = X.columns
        # On fit uniquement sur le train
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=cols)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=cols)
        
        # Sauvegarde pour les prochaines fois
        import joblib
        print("   -> Sauvegarde des splits pour la cohérence...")
        joblib.dump(X_train, os.path.join(self.processed_dir, 'X_train.joblib'))
        joblib.dump(X_test, os.path.join(self.processed_dir, 'X_test.joblib'))
        joblib.dump(X_val, os.path.join(self.processed_dir, 'X_val.joblib'))
        joblib.dump(y_train, os.path.join(self.processed_dir, 'y_train.joblib'))
        joblib.dump(y_test, os.path.join(self.processed_dir, 'y_test.joblib'))
        joblib.dump(y_val, os.path.join(self.processed_dir, 'y_val.joblib'))
        joblib.dump(labels_test, os.path.join(self.processed_dir, 'labels_test.joblib'))
        joblib.dump(labels_val, os.path.join(self.processed_dir, 'labels_val.joblib'))
        
        return X_train, X_test, X_val, y_train, y_test, y_val, labels_test, labels_val

    def _load_nsl_kdd(self):
        file_path = os.path.join(self.base_dir, 'data', 'nsl_kdd', 'KDDTrain+.txt')
        if not os.path.exists(file_path): raise FileNotFoundError(file_path)
        
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
        df['label'] = df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
        
        for col in ['protocol_type', 'service', 'flag']:
            df[col] = LabelEncoder().fit_transform(df[col])
            
        X = df.drop(['attack', 'level', 'label'], axis=1)
        y = df['label']
        return X, y, labels

    def _load_cic_ids(self):
        # Dossier contenant tous les CSV (Lundi, Mardi, Mercredi...)
        data_dir = os.path.join(self.base_dir, 'data', 'cic_ids2017')
        
        # On cherche tous les fichiers .csv dans le dossier
        all_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not all_files:
            raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {data_dir}. Vérifiez l'emplacement.")

        print(f"   -> Fichiers trouvés : {len(all_files)}")
        
        df_list = []
        # RATIO D'ÉCHANTILLONNAGE : 0.2 = On prend 20% de chaque fichier
        # Si vous avez 32Go de RAM, mettez 1.0. Si 16Go, mettez 0.2 ou 0.3.
        SAMPLE_RATIO = 1
        
        for filename in all_files:
            print(f"   -> Lecture de {os.path.basename(filename)}... (Ratio: {SAMPLE_RATIO})")
            try:
                # Lecture partielle pour économiser la mémoire
                # On utilise encoding='cp1252' car certains fichiers CIC ont des caractères bizarres
                temp_df = pd.read_csv(filename, encoding='cp1252', low_memory=False)
                
                # Nettoyage immédiat des noms de colonnes
                temp_df.columns = temp_df.columns.str.strip()
                
                # Sampling aléatoire
                if SAMPLE_RATIO < 1.0:
                    temp_df = temp_df.sample(frac=SAMPLE_RATIO, random_state=42)
                
                df_list.append(temp_df)
            except Exception as e:
                print(f"      [ERREUR] Impossible de lire {filename}: {e}")

        if not df_list:
            raise ValueError("Aucune donnée n'a pu être chargée.")

        # Fusion de tous les jours
        print("   -> Fusion des fichiers journaliers...")
        df = pd.concat(df_list, ignore_index=True)
        
        # Nettoyage Global
        print("   -> Nettoyage (Inf/NaN)...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Création des labels
        labels = df['Label']
        df['label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        # Sélection colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # On retire la cible numérique si elle est dedans
        if 'label' in numeric_cols: numeric_cols.remove('label')
        
        X = df[numeric_cols]
        y = df['label']
        
        print(f"   -> Dataset Final chargé : {len(X)} lignes.")
        return X, y, labels