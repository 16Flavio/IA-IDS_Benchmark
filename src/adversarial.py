import tensorflow as tf
import numpy as np
import pandas as pd

def generate_adversarial_pattern(model, input_data, target_label, epsilon=0.1):
    """
    Generate an adversarial example using the Fast Gradient Sign Method (FGSM).

    This function computes the gradient of the loss with respect to the input
    data and adds a small perturbation in the direction of the gradient to
    maximize the loss, effectively creating an adversarial example.

    Args:
        model: The Keras/TensorFlow model to attack.
        input_data: The input data (numpy array or DataFrame) of shape (1, n_features).
        target_label: The target label (typically the true label) to compute loss against.
        epsilon (float): The magnitude of the perturbation.

    Returns:
        tuple: A tuple containing:
            - perturbed_data (numpy.ndarray): The data with the added perturbation.
            - perturbation (numpy.ndarray): The perturbation that was added.
    """
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
        
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    target_tensor = tf.convert_to_tensor([target_label], dtype=tf.float32) 
    target_tensor = tf.reshape(target_tensor, (1, 1))

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model(input_tensor)
        loss = tf.keras.losses.binary_crossentropy(target_tensor, prediction)

    gradient = tape.gradient(loss, input_tensor)
    
    signed_grad = tf.sign(gradient)
    
    perturbation = signed_grad * epsilon
    perturbed_data = input_tensor + perturbation
    
    return perturbed_data.numpy(), perturbation.numpy()

def test_robustness(model_to_attack, models_to_test, X_sample, y_sample, epsilon=0.1):
    """
    Test the robustness of various models against adversarial attacks generated on a source model.

    This function performs a transferability test where adversarial examples are generated
    using a white-box model (`model_to_attack`) and then tested against other models
    (`models_to_test`) to see if they are also deceived.

    Args:
        model_to_attack: The source model (must be Keras/TensorFlow) used to generate attacks.
        models_to_test (dict): A dictionary of models to test, where keys are model names.
        X_sample (numpy.ndarray or pd.DataFrame): The sample of input data to generate attacks from.
        y_sample (numpy.ndarray or pd.Series): The true labels corresponding to X_sample.
        epsilon (float): The perturbation magnitude.

    Returns:
        dict: A dictionary containing success rates and total counts for each tested model.
              Format: {model_name: {"Success": count, "Total": count}}
    """
    results = {name: {"Success": 0, "Total": 0} for name in models_to_test.keys()}
    
    attack_indices = np.where(y_sample == 1)[0]
    
    if len(attack_indices) == 0:
        return results

    print(f"   [Robustness] Génération d'attaques adverses (Epsilon={epsilon}) sur source DL...")
    
    for idx in attack_indices:
        original_row = X_sample[idx:idx+1] 
        true_label = 1
        
        try:
             adv_example, _ = generate_adversarial_pattern(model_to_attack, original_row, true_label, epsilon)
        except Exception as e:
             continue

        for name, model in models_to_test.items():
            try:
                if hasattr(model, "predict"):
                    pred = model.predict(adv_example)
                    
                    if hasattr(pred, "values"): val = pred.values
                    elif hasattr(pred, "flatten"): val = pred.flatten()[0]
                    elif isinstance(pred, list): val = pred[0]
                    elif hasattr(pred, "item"): val = pred.item()
                    else: val = float(pred)
                    
                    is_detected = False
                         
                    if "Auto" in name or "Anomaly" in name:
                         is_detected = (val == 1)
                    else:
                         is_detected = (val == 1)

                    if not is_detected: 
                        results[name]["Success"] += 1
                    
                    results[name]["Total"] += 1
            except Exception as e:
                pass
            
    return results
