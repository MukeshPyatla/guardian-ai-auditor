import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from common.model_definition import TextComplianceModel, SensorAnomalyModel
from client_logic.he_utils import generate_global_paillier_keys, encrypt_value
import random
import os

# Ensure keys are generated (or retrieved from global scope)
# This ensures public_key_global is accessible for encryption.
# In a real system, keys would be loaded securely per client.
public_key, private_key = generate_global_paillier_keys()

def get_text_vectorizer():
    """Returns a pre-fitted TF-IDF vectorizer for consistent feature extraction."""
    # In a real FL scenario, this vectorizer (vocabulary) would be agreed upon globally.
    # Here, we'll create a simple one based on common terms.
    # For simplicity of demo and to avoid complex model sharing, we'll fit a basic one.
    common_phrases = [
        "terms conditions explained", "privacy policy understood", "no personal details",
        "bank account number requested", "shared customer data", "aggressive sales",
        "security protocol bypassed", "unencrypted log", "opt out options"
    ]
    vectorizer = TfidfVectorizer(max_features=100)
    vectorizer.fit(common_phrases)
    return vectorizer

GLOBAL_TEXT_VECTORIZER = get_text_vectorizer() # Initialize once

def preprocess_text_data(df):
    """Applies TF-IDF vectorization to text data using a global vectorizer."""
    X_text = GLOBAL_TEXT_VECTORIZER.transform(df['text']).toarray()
    # Convert 'compliant'/'non_compliant' to 0/1 for classification
    y_text = (df['true_compliance_status'] == 'non_compliant').astype(int) # 1 for non-compliant
    return X_text, y_text

def preprocess_sensor_data(df):
    """Standard scaling for sensor data."""
    scaler = StandardScaler()
    # Ensure sensor_value is 2D for scaler
    X_sensor = scaler.fit_transform(df[['sensor_value']])
    y_sensor = (df['true_anomaly_status'] == 'anomaly').astype(int) # 1 for anomaly
    return X_sensor, y_sensor

def get_local_insights(client_id, text_df, image_labels_df, sensor_df):
    """
    Performs local privacy-preserving feature extraction and anomaly/compliance detection.
    Returns encrypted/privacy-preserved insights.
    """
    print(f"Client {client_id}: Processing data and generating local insights...")

    # --- Text Modality ---
    X_text, y_text = preprocess_text_data(text_df)
    text_model = TextComplianceModel()
    if len(np.unique(y_text)) > 1: # Only fit if there are at least two classes
        text_model.fit(X_text, y_text)
        text_risk_score = text_model.predict_proba(X_text)[:, 1].mean() # Avg prob of non-compliant
        local_text_accuracy = text_model.model.score(X_text, y_text)
    else:
        text_risk_score = 0.0 # No non-compliant examples
        local_text_accuracy = 1.0 # If all are same class, perfect prediction
        print(f"Client {client_id}: Not enough classes in text data for robust text model training.")

    # --- Image Modality (Simplified: Using pre-generated labels as features for demo) ---
    # In a real scenario, this would involve feature extraction from image data itself (e.g., CNN embeddings)
    # We're using image_labels_df which contains 'anomaly'/'normal' directly for simplicity.
    if not image_labels_df.empty:
        image_anomaly_count = (image_labels_df['true_anomaly_status'] == 'anomaly').sum()
        image_risk_score = image_anomaly_count / len(image_labels_df)
        local_image_accuracy = 1.0 # Assuming perfect detection of pre-defined anomalies
    else:
        image_risk_score = 0.0
        local_image_accuracy = 1.0
        print(f"Client {client_id}: No image data for processing.")


    # --- Sensor Modality ---
    X_sensor, y_sensor = preprocess_sensor_data(sensor_df)
    sensor_model = SensorAnomalyModel()
    sensor_model.fit(X_sensor)
    sensor_predictions = sensor_model.predict(X_sensor)
    sensor_anomaly_rate = np.mean(sensor_predictions == -1) # -1 is anomaly for IsolationForest
    # We can't calculate a direct 'accuracy' for unsupervised anomaly detection easily
    # but we can compare to true labels if available for internal validation.
    local_sensor_accuracy = np.mean((sensor_predictions == -1) == y_sensor)


    # --- Encrypt and Return Local Insights ---
    # These are the numerical insights we'll conceptually share (after HE)
    # For demo, we'll only federate text model parameters for FL.
    # Other insights (risk scores) can be aggregated via conceptual HE sums.

    encrypted_text_risk = encrypt_value(float(text_risk_score), public_key)
    encrypted_image_risk = encrypt_value(float(image_risk_score), public_key)
    encrypted_sensor_risk = encrypt_value(float(sensor_anomaly_rate), public_key)

    return {
        "text_model_params": text_model.get_parameters(), # Parameters to be federated
        "encrypted_insights": {
            "text_risk": encrypted_text_risk,
            "image_risk": encrypted_image_risk,
            "sensor_risk": encrypted_sensor_risk
        },
        "true_metrics": { # For local validation and W&B logging (if not using encrypted metrics)
            "text_compliance_accuracy": local_text_accuracy,
            "image_anomaly_rate": image_risk_score, # Using this as accuracy proxy for demo
            "sensor_anomaly_accuracy": local_sensor_accuracy
        },
        "X_text": X_text, # Return for FL client to use
        "y_text": y_text  # Return for FL client to use
    }

def load_client_raw_data(client_id):
    """Loads raw synthetic data for a given client from local files."""
    base_path = os.path.join("data", "synthetic")
    text_df = pd.read_csv(os.path.join(base_path, f"{client_id}_text.csv"))
    image_labels_df = pd.read_csv(os.path.join(base_path, f"{client_id}_image_labels.csv"))
    sensor_df = pd.read_csv(os.path.join(base_path, f"{client_id}_sensor.csv"))
    return text_df, image_labels_df, sensor_df

def get_model_and_data_for_fl(client_id):
    """
    Loads client data, preprocesses, and returns a new model instance and
    the processed data (X, y) for FL client's fit method.
    """
    text_df, _, _ = load_client_raw_data(client_id)
    X_text, y_text = preprocess_text_data(text_df)

    # Check if there's enough data and distinct classes for training
    if X_text.shape[0] == 0 or len(np.unique(y_text)) < 2:
        print(f"Client {client_id}: Not enough data or classes for FL training. Skipping.")
        # Return a dummy model/data if cannot train
        return TextComplianceModel(), np.array([]), np.array([])

    model = TextComplianceModel()
    # Initial fit to ensure model has parameters set before FL round 1
    # This won't be saved, just initializes model weights
    try:
        model.fit(X_text, y_text)
    except ValueError as e:
        print(f"Client {client_id} initial fit failed: {e}. Returning empty model.")
        return TextComplianceModel(), np.array([]), np.array([])

    return model, X_text, y_text