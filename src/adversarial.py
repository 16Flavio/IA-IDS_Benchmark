import tensorflow as tf
import numpy as np
import pandas as pd

def generate_adversarial_pattern(model, input_data, target_label, epsilon=0.1):
    """
    Génère un exemple contradictoire (Adversarial Example) via FGSM.
    
    Args:
        model: Modèle Keras (TensorFlow)
        input_data: Numpy array ou DataFrame d'une seule ligne (shape (1, n_features))
        target_label: Label cible (int, ex: 1 pour attaque, 0 pour benign) - En réalité pour FGSM on veut maximiser la perte par rapport au VRAI label
        epsilon: Magnitude de la perturbation
    
    Returns:
        perturbed_data: Le paquet modifié
        perturbation: Le bruit ajouté
    """
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
        
    # Conversion en tenseur
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    target_tensor = tf.convert_to_tensor([target_label], dtype=tf.float32) # Shape (1,)
    target_tensor = tf.reshape(target_tensor, (1, 1))

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model(input_tensor)
        loss = tf.keras.losses.binary_crossentropy(target_tensor, prediction)

    # Calcul du gradient de la perte par rapport à l'entrée
    gradient = tape.gradient(loss, input_tensor)
    
    # Signe du gradient (FGSM)
    signed_grad = tf.sign(gradient)
    
    # Création de l'image perturbée (On monte la perte, donc on s'éloigne du bon label)
    # ATTENTION: Si on veut faire passer une attaque (1) pour du normal (0), 
    # la loss est calculée par rapport à 1. Si on maximise la loss, le modèle va prédire moins 1, donc plus 0.
    perturbation = signed_grad * epsilon
    perturbed_data = input_tensor + perturbation
    
    return perturbed_data.numpy(), perturbation.numpy()

def test_robustness(model_to_attack, models_to_test, X_sample, y_sample, epsilon=0.1):
    """
    Génère des attaques sur 'model_to_attack' (White-box, doit être Keras/TF)
    et teste si ces attaques trompent aussi 'models_to_test' (Black-box / Transferability).
    
    Args:
        model_to_attack: Modèle Keras (Deep Learning) pour générer les gradients
        models_to_test: Dictionnaire {"Nom": model} des modèles à évaluer
        X_sample: Echantillon d'attaques
        y_sample: Labels
    """
    results = {name: {"Success": 0, "Total": 0} for name in models_to_test.keys()}
    
    # On ne teste que sur les VRAIES attaques (y=1) qu'on veut camoufler
    attack_indices = np.where(y_sample == 1)[0]
    
    if len(attack_indices) == 0:
        return results

    print(f"   [Robustness] Génération d'attaques adverses (Epsilon={epsilon}) sur source DL...")
    
    for idx in attack_indices:
        # On prend une ligne
        original_row = X_sample[idx:idx+1] 
        true_label = 1
        
        # Génération de l'attaque (White-box sur le modèle DL)
        try:
             adv_example, _ = generate_adversarial_pattern(model_to_attack, original_row, true_label, epsilon)
        except Exception as e:
             # Si échec génération (ex: modèle non différentiable passé par erreur), on skip
             continue

        # Test sur TOUS les modèles (Transferabilité)
        for name, model in models_to_test.items():
            # Prédiction sur l'exemple adverse
            try:
                # Gestion ML vs DL vs AE
                if hasattr(model, "predict"):
                    pred = model.predict(adv_example)
                    
                    # Normalisation sortie
                    if hasattr(pred, "values"): val = pred.values
                    elif hasattr(pred, "flatten"): val = pred.flatten()[0]
                    elif isinstance(pred, list): val = pred[0]
                    elif hasattr(pred, "item"): val = pred.item()
                    else: val = float(pred)
                    
                    # LOGIQUE DE DÉCISION
                    # Pour AE : Si MSE < Seuil => C'est considéré comme NORMAL (0) => Attaque Réussie
                    # Pour classifieurs : Si Proba < 0.5 => C'est considéré comme NORMAL (0) => Attaque Réussie
                    
                    is_detected = False
                    
                    if "Auto" in name or "Anomaly" in name:
                         # Pour AE, predict retourne 1 si anomalie, 0 si normal (via méthode predict de method.py)
                         is_detected = (val == 1)
                         
                    if "Auto" in name or "Anomaly" in name:
                         # Pour AE, predict retourne 1 si anomalie, 0 si normal (via méthode predict de method.py)
                         is_detected = (val == 1)
                    else:
                         # Pour RF/DL/XGB, pred est 0 ou 1
                         is_detected = (val == 1)

                    if not is_detected: # L'attaque a réussi (classé comme bénin)
                        results[name]["Success"] += 1
                    
                    results[name]["Total"] += 1
            except Exception as e:
                pass
            
    return results
