import flwr as fl
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
import wandb
import random
import time

# Add parent directory to path to import common and client_logic modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from common.model_definition import TextComplianceModel
from client_logic.he_utils import generate_global_paillier_keys, decrypt_value, homomorphic_add_values, public_key_global, private_key_global
from client_logic.data_generator import generate_synthetic_text_data, save_client_data_locally

# Ensure keys are generated (or retrieved from global scope)
public_key, private_key = generate_global_paillier_keys()

# Define a simple evaluation function for the server
def get_eval_fn(test_data_path):
    """
    Returns a function that evaluates the global model on a public, non-sensitive test set.
    This simulates a public dataset used for overall model validation without client data access.
    """
    # Load or generate a small, separate test dataset for server evaluation.
    test_df = pd.read_csv(test_data_path)

    # Create a TF-IDF vectorizer that matches the client's feature space.
    test_vectorizer = TfidfVectorizer(max_features=100)
    common_phrases = [
        "terms conditions explained", "privacy policy understood", "no personal details",
        "bank account number requested", "shared customer data", "aggressive sales",
        "security protocol bypassed", "unencrypted log", "opt out options",
        "test sentence for compliance audit"
    ]
    test_vectorizer.fit(common_phrases + test_df['text'].tolist())

    X_test = test_vectorizer.transform(test_df['text']).toarray()
    y_test = (test_df['true_compliance_status'] == 'non_compliant').astype(int) # 1 for non-compliant

    # Initialize the global model once for evaluation purposes
    global_model_evaluator = TextComplianceModel()

    def evaluate(server_round, parameters, config):
        # Set the global model's parameters for evaluation
        num_features = X_test.shape[1]
        if not parameters: return 1.0, {"accuracy": 0.0}
        coef = np.array(parameters[:num_features]).reshape(1, -1)
        intercept = np.array(parameters[num_features:])
        global_model_evaluator.set_parameters({'coef': coef, 'intercept': intercept})

        preds = global_model_evaluator.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        loss = 1 - accuracy

        wandb.log({
            "server_round_accuracy": accuracy,
            "server_round_loss": loss,
            "round": server_round
        })
        print(f"Server Round {server_round} Global Accuracy (on public test set): {accuracy:.4f}")
        return float(loss), {"accuracy": float(accuracy)}
    return evaluate

def start_fl_server_main(num_rounds=3, num_clients=3):
    print("Starting Flower FL Server...")
    server_test_df = generate_synthetic_text_data(num_records=20, client_id="server_public_test", compliance_ratio=0.7)
    test_data_path = os.path.join("data", "synthetic", "server_public_test_text.csv")
    server_test_df.to_csv(test_data_path, index=False)
    print(f"Server public test data generated at: {test_data_path}")

    wandb.init(project="guardian-ai-fl", name="fl-server-run", reinit=True)
    print("FL Server W&B initialized.")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=get_eval_fn(test_data_path),
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    wandb.finish()

    print("\n--- Demonstrating Conceptual Homomorphic Aggregation on Server ---")
    client_encrypted_risks = []
    for i in range(num_clients):
        score = random.random() * 0.1 + (i * 0.05)
        client_encrypted_risks.append(public_key_global.encrypt(score))
        print(f"Simulated: Received encrypted risk score from client {i+1} (encrypted, not shown)")

    if client_encrypted_risks:
        total_encrypted_risk = client_encrypted_risks[0]
        for i in range(1, len(client_encrypted_risks)):
            total_encrypted_risk = homomorphic_add_values(total_encrypted_risk, client_encrypted_risks[i])
        decrypted_total_risk = decrypt_value(total_encrypted_risk, private_key_global)
        print(f"\nAggregated (Decrypted) Total Network Risk Score: {decrypted_total_risk:.4f}")
        print("This demonstrates that sensitive insights can be aggregated homomorphically across clients without decrypting individual contributions.")
    else:
        print("No encrypted insights to aggregate (check client setup).")

if __name__ == "__main__":
    start_fl_server_main(num_rounds=3, num_clients=3)