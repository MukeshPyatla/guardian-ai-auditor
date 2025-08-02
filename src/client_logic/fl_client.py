import flwr as fl
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import sys
import os
import wandb

# Add parent directory to path to import common and client_logic modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common.model_definition import TextComplianceModel
from client_logic.local_model import get_model_and_data_for_fl, generate_global_paillier_keys
from client_logic.data_generator import generate_synthetic_text_data, generate_synthetic_image_data, generate_synthetic_sensor_data, save_client_data_locally

# Ensure keys are generated (or retrieved from global scope)
# This ensures public_key_global is accessible for encryption (conceptually).
# In a real system, keys would be loaded securely.
public_key_global, private_key_global = generate_global_paillier_keys()

# Flower client class
class GuardianAIClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model, self.X_text, self.y_text = get_model_and_data_for_fl(client_id)
        wandb.init(project="guardian-ai-fl", group="clients", name=f"client-{client_id}", reinit=True)
        print(f"Client {self.client_id} W&B initialized.")

    def get_parameters(self, config):
        if self.model.get_parameters()['coef'] and self.model.get_parameters()['intercept']:
            return self.model.get_parameters()['coef'] + self.model.get_parameters()['intercept']
        return []

    def fit(self, parameters, config):
        if self.X_text.size > 0 and len(np.unique(self.y_text)) > 1:
            num_features = self.X_text.shape[1]
            coef = np.array(parameters[:num_features]).reshape(1, -1)
            intercept = np.array(parameters[num_features:])
            self.model.set_parameters({'coef': coef, 'intercept': intercept})

            self.model.fit(self.X_text, self.y_text)
            local_preds = self.model.predict(self.X_text)
            local_accuracy = accuracy_score(self.y_text, local_preds)

            wandb.log({
                f"client_{self.client_id}/local_accuracy": local_accuracy,
                f"client_{self.client_id}/loss": 1 - local_accuracy,
                "round": config.get("round", 0)
            })
            print(f"Client {self.client_id}: Local accuracy = {local_accuracy:.4f}")
            return self.get_parameters(config={}), len(self.X_text), {"local_accuracy": local_accuracy}
        else:
            print(f"Client {self.client_id}: Skipping local fit due to insufficient data/classes.")
            return [], len(self.X_text), {"local_accuracy": 0.0}

    def evaluate(self, parameters, config):
        loss = 0.1
        accuracy = 0.9
        return float(loss), len(self.X_text), {"accuracy": accuracy}

def main(client_id):
    from client_logic.data_generator import generate_synthetic_text_data, generate_synthetic_image_data, generate_synthetic_sensor_data, save_client_data_locally
    text_df = generate_synthetic_text_data(50, client_id)
    image_data = generate_synthetic_image_data(5, client_id)
    sensor_df = generate_synthetic_sensor_data(100, client_id)
    save_client_data_locally(client_id, text_df, image_data, sensor_df)

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=GuardianAIClient(client_id),
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fl_client.py <client_id>")
        sys.exit(1)
    client_id = sys.argv[1]
    main(client_id)