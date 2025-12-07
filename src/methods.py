import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import joblib
import os

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

# MÉTHODE TRADITIONNELLE (Basée sur des seuils - Rule Based)
class RuleBasedDetector(BaseEstimator):
    def fit(self, X, y=None):
        pass # Pas d'entrainement pour des règles statiques
    
    def predict(self, X):
        # Exemple de règle naïve : Si beaucoup de trafic sortant -> Attaque
        # Dans un vrai projet, ces règles viendraient de SNORT.
        preds = []
        for _, row in X.iterrows():
            # Si src_bytes est énorme ou durée très longue (règle arbitraire pour l'exemple)
            if row.get('src_bytes', 0) > 10000 or row.get('Flow Duration', 0) > 50000:
                preds.append(1)
            else:
                preds.append(0)
        return preds
    
    def save_model(self, path):
        pass
    
    def load_model(self, path):
        pass

# MACHINE LEARNING (Random Forest)
class MLDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def train(self, X_train, y_train):
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

# DEEP LEARNING (Simple MLP)
class DLDetector:
    def __init__(self, input_shape=None):
        if input_shape:
            self.model = Sequential([
                Input(shape=(input_shape,)),
                
                # Couche 1 : Plus large + Normalisation
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                
                # Couche 2
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                
                # Couche 3
                Dense(32, activation='relu'),
                
                # Sortie
                Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            self.model = None

    def train(self, X_train, y_train):
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        self.model.fit(
            X_train, y_train, 
            epochs=500, 
            batch_size=1024,
            callbacks=[early_stop],
            verbose=1
        )
        
    def predict(self, X):
        return (self.model.predict(X, batch_size=1024) > 0.5).astype("int32")
    
    def predict_proba(self, X):
        return self.model.predict(X, batch_size=1024)
    
    def save_model(self, path):
        self.model.save(path) # Sauvegarde au format .keras ou .h5
        print(f"Modèle DL sauvegardé : {path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.model = load_model(path)
            print(f"Modèle DL chargé depuis : {path}")
        else:
            print(f"Erreur : Fichier modèle introuvable {path}")

# HYBRIDE (DL + LOGIQUE DE SÉCURITÉ)
class HybridDetector:
    def __init__(self, dl_model):
        self.dl_model = dl_model
        
    def predict(self, X):
        probabilities = self.dl_model.predict_proba(X)
        final_preds = []
        
        is_cic = 'Destination Port' in X.columns
        
        for i, prob in enumerate(probabilities):
            p = prob[0]
            
            # Zone de Confiance IA
            if p > 0.75: # J'ai baissé un peu le seuil pour faire confiance plus vite
                final_preds.append(1)
            elif p < 0.25:
                final_preds.append(0)
            
            # Zone d'Incertitude (IA perdue) -> RÈGLES EXPERTES
            else:
                row = X.iloc[i]
                
                if is_cic:
                    if row['Flow Duration'] > 0.5 or row['Total Fwd Packets'] > 1.0:
                        final_preds.append(1) # Probable DoS
                    else:
                        final_preds.append(0)
                        
                else: # NSL-KDD
                    if row.get('dst_host_count', 0) > 0.5 or row.get('src_bytes', 0) > 1.0:
                        final_preds.append(1)
                    else:
                        final_preds.append(0)
                    
        return final_preds