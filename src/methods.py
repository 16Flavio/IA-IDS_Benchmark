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

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost non installé. Le détecteur XGBoost ne fonctionnera pas.")

class RuleBasedDetector(BaseEstimator):
    """
    Détecteur basé sur des règles statistiques et une liste noire d'IPs.
    """
    def __init__(self):
        """
        Initialise le détecteur avec des seuils vides et une liste de surveillance.
        """
        self.thresholds = {}
        self.features_to_monitor = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
            'Total Length of Fwd Packets', 'Fwd Packet Length Max',
            'src_bytes', 'dst_bytes', 'count', 'srv_count'
        ]
        self.blocked_ips = set()

    def update_blocklist(self, ip_address):
        """
        Ajoute une adresse IP à la liste noire.

        Args:
            ip_address (str): L'adresse IP à bloquer.
        """
        self.blocked_ips.add(ip_address)

    def is_blocked(self, ip_address):
        """
        Vérifie si une adresse IP est présente dans la liste noire.

        Args:
            ip_address (str): L'adresse IP à vérifier.

        Returns:
            bool: True si l'IP est bloquée, False sinon.
        """
        return ip_address in self.blocked_ips

    def fit(self, X, y):
        """
        Calcule les seuils statistiques basés sur le trafic bénin.

        Args:
            X (pd.DataFrame): Les données d'entraînement.
            y (array-like): Les labels associés (0 pour bénin).
        """
        print("   [RuleBased] Calcul des seuils statistiques sur le trafic bénin...")
        if not isinstance(X, pd.DataFrame):
            print("   [RuleBased] Attention: Input n'est pas un DataFrame.")
            return

        try:
            X = X.reset_index(drop=True)
            if hasattr(y, 'reset_index'):
                y = y.reset_index(drop=True)
            elif hasattr(y, 'values'): 
                y = pd.Series(y.values)
            else:
                y = pd.Series(y)
                
            X_benign = X[y == 0]
            
            for feature in self.features_to_monitor:
                if feature in X_benign.columns:
                    limit = X_benign[feature].quantile(0.999) 
                    self.thresholds[feature] = limit * 1.1 
        except Exception as e:
            print(f"   [RuleBased] ERREUR lors du fit : {e}")

    def predict(self, X):
        """
        Prédit si le trafic est normal ou une anomalie en fonction des seuils.

        Args:
            X (pd.DataFrame): Les données à analyser.

        Returns:
            np.array: Un tableau de prédictions (0 ou 1).
        """
        preds = []
        
        if not self.thresholds:
            return np.zeros(len(X), dtype=int)

        if not isinstance(X, pd.DataFrame):
             return np.zeros(len(X), dtype=int)

        for _, row in X.iterrows():
            is_anomaly = False
            for feature, limit in self.thresholds.items():
                if feature in row and row[feature] > limit:
                    is_anomaly = True
                    break
            
            preds.append(1 if is_anomaly else 0)
            
        return np.array(preds)
    
    def save_model(self, path):
        """
        Sauvegarde les seuils et la liste noire dans un fichier.

        Args:
            path (str): Le chemin du fichier de sauvegarde.
        """
        data = {
            'thresholds': self.thresholds,
            'blocked_ips': self.blocked_ips
        }
        joblib.dump(data, path)
        print(f"Règles statistiques sauvegardées : {path}")
    
    def load_model(self, path):
        """
        Charge les seuils et la liste noire depuis un fichier.

        Args:
            path (str): Le chemin du fichier à charger.
        """
        if os.path.exists(path):
            data = joblib.load(path)
            if isinstance(data, dict) and 'thresholds' in data:
                self.thresholds = data['thresholds']
                self.blocked_ips = data.get('blocked_ips', set())
            else:
                self.thresholds = data
                self.blocked_ips = set()
                
            print(f"Règles statistiques chargées ({len(self.thresholds)} seuils).")
        else:
            print("Erreur : Fichier règles introuvable.")


class MLDetector:
    """
    Détecteur basé sur le Machine Learning (Random Forest).
    """
    def __init__(self):
        """
        Initialise le modèle Random Forest.
        """
        self.model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entraîne le modèle Random Forest.

        Args:
            X_train (array-like): Données d'entraînement.
            y_train (array-like): Labels d'entraînement.
            X_val (array-like, optional): Données de validation (non utilisé pour RF mais gardé pour uniformité).
            y_val (array-like, optional): Labels de validation.
        """
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """
        Prédit les classes pour les données fournies.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Retourne les probabilités des classes prédites.
        """
        return self.model.predict_proba(X)

    def save_model(self, path):
        """
        Sauvegarde le modèle sur le disque.
        """
        joblib.dump(self.model, path)
        print(f"Modèle ML sauvegardé : {path}")

    def load_model(self, path):
        """
        Charge le modèle depuis le disque.
        """
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Modèle ML chargé depuis : {path}")
        else:
            print("Erreur : Fichier modèle introuvable.")

class XGBoostDetector:
    """
    Détecteur basé sur XGBoost (eXtreme Gradient Boosting).
    """
    def __init__(self):
        """
        Initialise le modèle XGBoost si la librairie est disponible.
        """
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
        """
        Entraîne le modèle XGBoost avec monitoring possible sur le set de validation.
        """
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

    def predict(self, X):
        """
        Prédit les classes.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Retourne les probabilités.
        """
        return self.model.predict_proba(X)

    def save_model(self, path):
        """
        Sauvegarde le modèle.
        """
        joblib.dump(self.model, path)
        print(f"Modèle XGBoost sauvegardé : {path}")

    def load_model(self, path):
        """
        Charge le modèle.
        """
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Modèle XGBoost chargé depuis : {path}")
        else:
            print("Erreur : Fichier modèle introuvable.")

class DLDetector:
    """
    Détecteur basé sur le Deep Learning (Réseau de Neurones Multi-Couches).
    """
    def __init__(self, input_shape=None):
        """
        Initialise l'architecture du modèle si input_shape est fourni.
        """
        if input_shape:
            self.model = Sequential([
                Input(shape=(input_shape,)),
                
                Dense(256),
                BatchNormalization(),
                Activation('relu'),
                Dropout(0.4),
                
                Dense(128),
                BatchNormalization(),
                Activation('relu'),
                Dropout(0.3),
                
                Dense(64),
                BatchNormalization(),
                Activation('relu'),
                Dropout(0.2),

                Dense(32),
                BatchNormalization(),
                Activation('relu'),
                
                Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entraîne le modèle avec Early Stopping et réduction du learning rate.
        """
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
        """
        Prédit les classes (0 ou 1) en fonction d'un seuil de 0.5.
        """
        return (self.model.predict(X, batch_size=1024, verbose=0) > 0.5).astype("int32")
    
    def predict_proba(self, X):
        """
        Retourne la probabilité d'anomalie/attaque.
        """
        return self.model.predict(X, batch_size=1024, verbose=0)
    
    def save_model(self, path):
        """
        Sauvegarde le modèle complet (architecture + poids).
        """
        self.model.save(path) 
        print(f"Modèle DL sauvegardé : {path}")

    def load_model(self, path):
        """
        Charge un modèle sauvegardé.
        """
        if os.path.exists(path):
            self.model = load_model(path)
            print(f"Modèle DL chargé depuis : {path}")
        else:
            print(f"Erreur : Fichier modèle introuvable {path}")

class AnomalyDetector:
    """
    Détecteur d'anomalies basé sur un Auto-Encoder (Apprentissage non-supervisé / Zero-Day).
    """
    def __init__(self, input_shape=None):
        """
        Initialise l'Auto-Encoder.
        """
        if input_shape:
            input_layer = Input(shape=(input_shape,))
            
            encoder = Dense(64, activation="relu")(input_layer)
            encoder = Dropout(0.2)(encoder) 
            
            encoder = Dense(32, activation="relu")(encoder)
            encoder = Dense(16, activation="relu")(encoder)
            
            decoder = Dense(32, activation="relu")(encoder)
            decoder = Dense(64, activation="relu")(decoder)
            output_layer = Dense(input_shape, activation="linear")(decoder) 
            
            self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
            
            self.model.compile(optimizer='adam', loss='mae') 
            self.threshold = None
        else:
            self.model = None
            self.threshold = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entraîne l'Auto-Encoder uniquement sur le trafic bénin.
        """
        print("   [AutoEncoder] Entraînement sur trafic bénin uniquement...")
        
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.reset_index(drop=True)
            if hasattr(y_train, 'reset_index'):
                y_train = y_train.reset_index(drop=True)
            elif hasattr(y_train, 'values'):
                y_train = pd.Series(y_train.values)
        
        X_benign = X_train[y_train == 0]
        
        self.model.fit(
            X_benign, X_benign,
            epochs=50,
            batch_size=256,
            shuffle=True,
            validation_split=0.1,
            verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )
        
        if X_val is not None and y_val is not None:
            self.find_optimal_threshold(X_val, y_val)
        else:
            print("   [AutoEncoder] Pas de set de validation fourni. Calcul seuil statistique sur Train...")
            reconstructions = self.model.predict(X_benign, verbose=0)
            mse = np.mean(np.power(X_benign - reconstructions, 2), axis=1)
            self.threshold = np.percentile(mse, 95)
            print(f"   [AutoEncoder] Seuil (95th percentile) : {self.threshold:.6f}")

    def find_optimal_threshold(self, X, y):
        """
        Calcule le seuil de d'erreur de reconstruction optimal en maximisant le F1-Score sur le set de validation.
        """
        print("   [AutoEncoder] Optimisation du seuil sur le set de Validation...")
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        precisions, recalls, thresholds = precision_recall_curve(y, mse)
        
        numerator = 2 * (precisions * recalls)
        denominator = (precisions + recalls)
        f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        self.threshold = best_threshold
        print(f"   [AutoEncoder] Seuil Optimal trouvé : {self.threshold:.6f} (Best Val F1: {best_f1:.4f})")

    def predict(self, X):
        """
        Prédit si le trafic est une anomalie (Erreur reconstruction > Seuil).
        """
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        return (mse > self.threshold).astype(int)
    
    def predict_proba(self, X):
        """
        Retourne l'erreur de reconstruction (MSE), qui sert de score d'anomalie.
        """
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        return mse 

    def save_model(self, path):
        """
        Sauvegarde le modèle et le seuil optimisé.
        """
        base_dir = os.path.dirname(path)
        base_name = os.path.basename(path).replace('.keras', '')
        
        keras_path = os.path.join(base_dir, f"{base_name}_model.keras")
        meta_path = os.path.join(base_dir, f"{base_name}_meta.joblib")
        
        self.model.save(keras_path)
        joblib.dump(self.threshold, meta_path)
        print(f"AutoEncoder sauvegardé : {keras_path} (Seuil: {self.threshold:.4f})")

    def load_model(self, path):
        """
        Charge le modèle et le seuil.
        """
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


class HybridDetector:
    """
    Système hybride combinant un modèle Deep Learning et un système basé sur des règles.
    """
    def __init__(self, dl_model, rb_model=None):
        """
        Initialise le détecteur hybride.
        """
        self.dl_model = dl_model
        self.rb_model = rb_model 
        
    def predict(self, X):
        """
        Prédit la classe en combinant les sorties du DL et du RuleBased.
        Utilise le RuleBasedDetector si le DL est incertain.
        """
        probabilities = self.dl_model.predict_proba(X)
        final_preds = []
        
        if self.rb_model:
            rb_preds = self.rb_model.predict(X) 
        else:
            rb_preds = np.zeros(len(X), dtype=int)

        for i, prob in enumerate(probabilities):
            p = prob[0]
            
            if p > 0.80:
                final_preds.append(1) 
            elif p < 0.20:
                final_preds.append(0) 
            
            else:
                if rb_preds[i] == 1:
                    final_preds.append(1)
                else:
                    final_preds.append(1 if p >= 0.5 else 0)
                    
        return np.array(final_preds)
    
    def predict_proba(self, X):
        """
        Retourne la probabilité brute du modèle Deep Learning.
        """
        return self.dl_model.predict(X)