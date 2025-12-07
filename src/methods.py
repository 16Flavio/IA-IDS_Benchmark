import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

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

# MACHINE LEARNING (Random Forest)
class MLDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)

# DEEP LEARNING (Simple MLP)
class DLDetector:
    def __init__(self, input_shape):
        self.model = Sequential([
            Input(shape=(input_shape,)),
            
            # Ensuite tes couches habituelles
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def train(self, X_train, y_train):
        # Conversion explicite en tenseurs TF (aide parfois à forcer l'usage GPU)
        # Mais Keras le fait souvent tout seul.
        
        self.model.fit(
            X_train, y_train, 
            epochs=500,          # On peut augmenter les epochs car le GPU est rapide
            batch_size=1024,    # IMPORTANT : Avec un GPU, on augmente le Batch Size !
                                # 32 est trop petit, le GPU s'ennuie. 
                                # 1024 ou 2048 est mieux pour paralléliser.
            verbose=1
        )
        
    def predict(self, X):
        return (self.model.predict(X, batch_size=1024) > 0.5).astype("int32")
    
    def predict_proba(self, X):
        return self.model.predict(X, batch_size=1024)
    
# HYBRIDE (DL + LOGIQUE DE SÉCURITÉ)
class HybridDetector:
    def __init__(self, dl_model):
        self.dl_model = dl_model
        
    def predict(self, X):
        # Prédiction brute du Deep Learning
        probabilities = self.dl_model.predict_proba(X)
        final_preds = []
        
        for i, prob in enumerate(probabilities):
            p = prob[0]
            # Logique Hybride :
            # Si l'IA est sûre (> 80% ou < 20%), on lui fait confiance.
            if p > 0.8:
                final_preds.append(1)
            elif p < 0.2:
                final_preds.append(0)
            else:
                # ZONE D'INCERTITUDE (20-80%) : On applique une règle logique stricte ("Fail-safe")
                # Ex: Si incertain mais trafic vers un port sensible (ex: 22 SSH), on bloque par précaution.
                row = X.iloc[i]
                if row.get('dst_host_count', 0) > 50: # Règle de sécurité
                    final_preds.append(1) # On force l'attaque (Paranoïaque)
                else:
                    final_preds.append(0) # On laisse passer
                    
        return final_preds