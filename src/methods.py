import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_recall_curve
import joblib
import os
import numpy as np
import pandas as pd

# --- CONFIGURATION GPU ---
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU DÉTECTÉ ET ACTIVÉ : {len(gpus)} GPU(s) trouvé(s)")
    else:
        print("AUCUN GPU DÉTECTÉ - Le Deep Learning tournera sur le CPU (plus lent)")
except RuntimeError as e:
    print(f"Erreur configuration GPU : {e}")

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost non installé. Le détecteur XGBoost ne fonctionnera pas.")

# MÉTHODE TRADITIONNELLE (Basée sur des seuils statistiques - Anomaly Detection + Defense Active)
class RuleBasedDetector(BaseEstimator):
    def __init__(self):
        self.thresholds = {}
        self.features_to_monitor = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
            'Total Length of Fwd Packets', 'Fwd Packet Length Max',
            'src_bytes', 'dst_bytes', 'count', 'srv_count'
        ]
        # Active Defense : Liste noire d'IPs
        self.blocked_ips = set()

    def update_blocklist(self, ip_address):
        """Ajoute une IP à la liste noire."""
        self.blocked_ips.add(ip_address)
        # print(f"   [FIREWALL] IP Bannnie : {ip_address}")

    def is_blocked(self, ip_address):
        """Vérifie si une IP est bloquée."""
        return ip_address in self.blocked_ips

    def fit(self, X, y):
        """
        Apprend le profil 'Normal' du trafic.
        On ne garde que le trafic Bénin (y==0) pour calculer les seuils.
        """
        print("   [RuleBased] Calcul des seuils statistiques sur le trafic bénin...")
        # On s'assure de travailler sur un DataFrame
        if not isinstance(X, pd.DataFrame):
            print("   [RuleBased] Attention: Input n'est pas un DataFrame.")
            return

        # Alignement des index pour éviter l'erreur "Unalignable boolean Series"
        try:
            # On reset les index pour être sûr
            X = X.reset_index(drop=True)
            if hasattr(y, 'reset_index'):
                y = y.reset_index(drop=True)
            elif hasattr(y, 'values'): # If it's a series wrapped
                y = pd.Series(y.values)
            else:
                y = pd.Series(y)
                
            # Filtrer uniquement le trafic bénin
            # y doit être un booléen ou 0/1. On assume 0 = Benign
            X_benign = X[y == 0]
            
            for feature in self.features_to_monitor:
                if feature in X_benign.columns:
                    limit = X_benign[feature].quantile(0.999) 
                    self.thresholds[feature] = limit * 1.1 
        except Exception as e:
            print(f"   [RuleBased] ERREUR lors du fit : {e}")

    def predict(self, X):
        preds = []
        
        if not self.thresholds:
            # Si pas de seuils (pas de fit), on renvoie tout à 0 (Safe)
            return np.zeros(len(X), dtype=int)

        if not isinstance(X, pd.DataFrame):
             return np.zeros(len(X), dtype=int)

        for _, row in X.iterrows():
            is_anomaly = False
            # Une requête est une anomalie si elle explose l'un des compteurs
            for feature, limit in self.thresholds.items():
                if feature in row and row[feature] > limit:
                    is_anomaly = True
                    break
            
            preds.append(1 if is_anomaly else 0)
            
        return np.array(preds)
    
    def save_model(self, path):
        # On sauvegarde aussi la blocklist, pourquoi pas (bien que ce soit dynamique)
        data = {
            'thresholds': self.thresholds,
            'blocked_ips': self.blocked_ips
        }
        joblib.dump(data, path)
        print(f"Règles statistiques sauvegardées : {path}")
    
    def load_model(self, path):
        if os.path.exists(path):
            data = joblib.load(path)
            # Gestion rétro-compatibilité (si ancien fichier contenant juste dict)
            if isinstance(data, dict) and 'thresholds' in data:
                self.thresholds = data['thresholds']
                self.blocked_ips = data.get('blocked_ips', set())
            else:
                self.thresholds = data
                self.blocked_ips = set()
                
            print(f"Règles statistiques chargées ({len(self.thresholds)} seuils).")
        else:
            print("Erreur : Fichier règles introuvable.")


# MACHINE LEARNING (Random Forest)
class MLDetector:
    def __init__(self):
        # Optimisation : n_jobs=-1 utilise tous les cœurs
        self.model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # RF n'utilise pas de validation set pour le training direct, mais on garde la signature
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save_model(self, path):
        joblib.dump(self.model, path)
        print(f"Modèle ML sauvegardé : {path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Modèle ML chargé depuis : {path}")
        else:
            print("Erreur : Fichier modèle introuvable.")

# XGBOOST DETECTOR
class XGBoostDetector:
    def __init__(self):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost n'est pas installé.")
        
        self.model = XGBClassifier(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=10, 
            subsample=0.8, 
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            eval_metric='logloss'
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save_model(self, path):
        joblib.dump(self.model, path)
        print(f"Modèle XGBoost sauvegardé : {path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Modèle XGBoost chargé depuis : {path}")
        else:
            print("Erreur : Fichier modèle introuvable.")

# DEEP LEARNING (Enhanced MLP)
class DLDetector:
    def __init__(self, input_shape=None):
        if input_shape:
            self.model = Sequential([
                Input(shape=(input_shape,)),
                
                # Couche 1
                Dense(256),
                BatchNormalization(),
                Activation('relu'),
                Dropout(0.4),
                
                # Couche 2
                Dense(128),
                BatchNormalization(),
                Activation('relu'),
                Dropout(0.3),
                
                # Couche 3
                Dense(64),
                BatchNormalization(),
                Activation('relu'),
                Dropout(0.2),

                # Couche 4
                Dense(32),
                BatchNormalization(),
                Activation('relu'),
                
                # Sortie
                Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        val_data = None
        if X_val is not None and y_val is not None:
            val_data = (X_val, y_val)
        
        self.model.fit(
            X_train, y_train, 
            validation_data=val_data,
            epochs=100, 
            batch_size=512,
            callbacks=callbacks,
            verbose=0
        )
        
    def predict(self, X):
        return (self.model.predict(X, batch_size=1024, verbose=0) > 0.5).astype("int32")
    
    def predict_proba(self, X):
        return self.model.predict(X, batch_size=1024, verbose=0)
    
    def save_model(self, path):
        self.model.save(path) # Sauvegarde au format .keras ou .h5
        print(f"Modèle DL sauvegardé : {path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.model = load_model(path)
            print(f"Modèle DL chargé depuis : {path}")
        else:
            print(f"Erreur : Fichier modèle introuvable {path}")

# AUTO-ENCODER (DÉTECTION D'ANOMALIES NON-SUPERVISÉE - ZERO DAY)
class AnomalyDetector:
    def __init__(self, input_shape=None):
        if input_shape:
            # Encoder
            input_layer = Input(shape=(input_shape,))
            
            # --- MODIFICATION : Ajout de Dropout et architecture plus robuste ---
            encoder = Dense(64, activation="relu")(input_layer)
            encoder = Dropout(0.2)(encoder) # Evite d'apprendre le bruit
            
            encoder = Dense(32, activation="relu")(encoder)
            encoder = Dense(16, activation="relu")(encoder)
            
            # Decoder
            decoder = Dense(32, activation="relu")(encoder)
            decoder = Dense(64, activation="relu")(decoder)
            output_layer = Dense(input_shape, activation="linear")(decoder) # Reconstruction
            
            self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
            
            # --- MODIFICATION : Utilisation de MAE pour l'entrainement (plus robuste aux outliers) ---
            self.model.compile(optimizer='adam', loss='mae') 
            self.threshold = None
        else:
            self.model = None
            self.threshold = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entraînement uniquement sur le trafic BÉNIN (y=0).
        Le modèle apprend à reconstruire le trafic normal.
        """
        print("   [AutoEncoder] Entraînement sur trafic bénin uniquement...")
        
        # Alignement index pour filtrage
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.reset_index(drop=True)
            if hasattr(y_train, 'reset_index'):
                y_train = y_train.reset_index(drop=True)
            elif hasattr(y_train, 'values'):
                y_train = pd.Series(y_train.values)
        
        # Filtre Benign
        X_benign = X_train[y_train == 0]
        
        # Training
        self.model.fit(
            X_benign, X_benign,
            epochs=50,
            batch_size=256,
            shuffle=True,
            validation_split=0.1,
            verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )
        
        # --- MODIFICATION : Calcul optimisé du seuil sur la Validation ---
        if X_val is not None and y_val is not None:
            self.find_optimal_threshold(X_val, y_val)
        else:
            # Fallback : Méthode statistique sur le train (moins précis)
            print("   [AutoEncoder] Pas de set de validation fourni. Calcul seuil statistique sur Train...")
            reconstructions = self.model.predict(X_benign, verbose=0)
            mse = np.mean(np.power(X_benign - reconstructions, 2), axis=1)
            self.threshold = np.percentile(mse, 95)
            print(f"   [AutoEncoder] Seuil (95th percentile) : {self.threshold:.6f}")

    def find_optimal_threshold(self, X, y):
        """
        Trouve le seuil qui maximise le F1-Score sur un set contenant des attaques.
        """
        print("   [AutoEncoder] Optimisation du seuil sur le set de Validation...")
        # 1. Obtenir les erreurs de reconstruction (Score d'anomalie)
        # Note : On garde le MSE pour le score d'anomalie car il pénalise + les grosses erreurs
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        # 2. Calculer les précisions/rappels pour tous les seuils possibles
        precisions, recalls, thresholds = precision_recall_curve(y, mse)
        
        # 3. Calculer le F1 Score pour chaque seuil
        # On évite la division par zéro
        numerator = 2 * (precisions * recalls)
        denominator = (precisions + recalls)
        f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        # 4. Trouver l'index du meilleur F1
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        self.threshold = best_threshold
        print(f"   [AutoEncoder] Seuil Optimal trouvé : {self.threshold:.6f} (Best Val F1: {best_f1:.4f})")

    def predict(self, X):
        """
        Si Erreur Reconstruction > Seuil => Anomalie (1)
        Sinon => Normal (0)
        """
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        # Si mse > seuil, c'est une anomalie (1)
        return (mse > self.threshold).astype(int)
    
    def predict_proba(self, X):
        # Pour ROC Curve, on peut renvoyer la MSE normalisée ou brute
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        return mse # Plus c'est grand, plus c'est une anomalie

    def save_model(self, path):
        # On sauvegarde le modèle Keras + le seuil
        base_dir = os.path.dirname(path)
        base_name = os.path.basename(path).replace('.keras', '')
        
        keras_path = os.path.join(base_dir, f"{base_name}_model.keras")
        meta_path = os.path.join(base_dir, f"{base_name}_meta.joblib")
        
        self.model.save(keras_path)
        joblib.dump(self.threshold, meta_path)
        print(f"AutoEncoder sauvegardé : {keras_path} (Seuil: {self.threshold:.4f})")

    def load_model(self, path):
        # Path est générique, on déduit les 2 fichiers
        base_dir = os.path.dirname(path)
        base_name = os.path.basename(path).replace('.keras', '')
        
        keras_path = os.path.join(base_dir, f"{base_name}_model.keras")
        meta_path = os.path.join(base_dir, f"{base_name}_meta.joblib")
        
        if os.path.exists(keras_path) and os.path.exists(meta_path):
            self.model = load_model(keras_path)
            self.threshold = joblib.load(meta_path)
            print(f"AutoEncoder chargé. Seuil: {self.threshold:.4f}")
        else:
            print(f"Erreur : Fichiers AutoEncoder introuvables ({keras_path})")


# HYBRIDE (DL + LOGIQUE DE SÉCURITÉ)
class HybridDetector:
    def __init__(self, dl_model, rb_model=None):
        self.dl_model = dl_model
        self.rb_model = rb_model # Le modèle basé sur les règles statistiques
        
    def predict(self, X):
        probabilities = self.dl_model.predict_proba(X)
        final_preds = []
        
        # Pré-calcul des règles si disponible
        if self.rb_model:
            rb_preds = self.rb_model.predict(X) 
        else:
            rb_preds = np.zeros(len(X), dtype=int)

        for i, prob in enumerate(probabilities):
            p = prob[0]
            
            # 1. Confiance IA Élevée
            if p > 0.80:
                final_preds.append(1) # Attaque quasi-sûre
            elif p < 0.20:
                final_preds.append(0) # Benign quasi-sûr
            
            # 2. IA Incertaine (entre 0.20 et 0.80) -> On demande aux règles
            else:
                # Si le modèle statistique dit "Anomalie", on le croit
                if rb_preds[i] == 1:
                    final_preds.append(1)
                else:
                    # Sinon, on se rabat sur la décision de l'IA (seuil 0.5)
                    final_preds.append(1 if p >= 0.5 else 0)
                    
        return np.array(final_preds)
    
    def predict_proba(self, X):
        return self.dl_model.predict(X)