import pandas as pd
import numpy as np
from faker import Faker
import random
from PIL import Image
import os

fake = Faker()

def generate_synthetic_text_data(num_records=100, client_id="client_0", compliance_ratio=0.8):
    """Generates synthetic chat log data with compliance/non-compliance phrases."""
    data = []
    compliance_phrases = [
        "All terms and conditions were clearly explained.",
        "Customer confirmed understanding of privacy policy.",
        "No personal financial details were requested.",
        "Ensured data collection consent was verbalized.",
        "Standard operating procedure was followed precisely.",
        "Information security protocols were strictly adhered to."
    ]
    non_compliance_phrases = [
        "Asked for customer's full bank account number.",
        "Shared customer data without consent.",
        "Used aggressive sales tactics.",
        "Did not disclose all fees upfront.",
        "Bypassed a required security step for speed.",
        "Recorded sensitive information in unencrypted log.",
        "Mentioned competitor names during the call.",
        "Did not offer opt-out options clearly."
    ]
    for i in range(num_records):
        user_name = fake.name()
        conversation_id = f"conv_{client_id}_{i}"
        is_compliant = random.random() < compliance_ratio
        if is_compliant:
            phrase = random.choice(compliance_phrases)
            compliance_status = "compliant"
        else:
            phrase = random.choice(non_compliance_phrases)
            compliance_status = "non_compliant"

        data.append({
            "client_id": client_id,
            "conversation_id": conversation_id,
            "timestamp": fake.date_time_this_year(),
            "speaker": random.choice(["Agent", "Customer"]),
            "text": f"[{user_name}]: {phrase}",
            "true_compliance_status": compliance_status
        })
    df = pd.DataFrame(data)
    return df

def generate_synthetic_image_data(num_images=10, client_id="client_0"):
    """Generates synthetic images (as numpy arrays) with simple 'anomalies'."""
    images_data = []
    for i in range(num_images):
        img_array = np.full((28, 28), 255, dtype=np.uint8)
        is_anomaly = random.random() < 0.2
        label = "normal"
        if is_anomaly:
            img_array[10:18, 10:18] = 0
            label = "anomaly"
        images_data.append((img_array, label))
    return images_data

def generate_synthetic_sensor_data(num_points=200, client_id="client_0"):
    """Generates synthetic time series sensor data with anomalies."""
    time = np.arange(num_points)
    signal = 10 * np.sin(time / 10) + time * 0.1
    noise = np.random.normal(0, 0.5, num_points)
    data = signal + noise

    anomaly_indices = random.sample(range(num_points), int(num_points * 0.05))
    labels = np.full(num_points, "normal", dtype=object)
    for idx in anomaly_indices:
        data[idx] += random.choice([-1, 1]) * np.random.uniform(5, 10)
        labels[idx] = "anomaly"

    df = pd.DataFrame({
        "client_id": client_id,
        "timestamp": pd.to_datetime(pd.date_range("2024-01-01", periods=num_points, freq="H")),
        "sensor_value": data,
        "true_anomaly_status": labels
    })
    return df

def save_client_data_locally(client_id, text_df, image_data, sensor_df):
    """Saves generated synthetic data to the client's local directory (simulated)."""
    output_dir = os.path.join("data", "synthetic")
    os.makedirs(output_dir, exist_ok=True)
    text_df.to_csv(os.path.join(output_dir, f"{client_id}_text.csv"), index=False)
    sensor_df.to_csv(os.path.join(output_dir, f"{client_id}_sensor.csv"), index=False)
    image_labels_df = pd.DataFrame([{"client_id": client_id, "image_id": i, "true_anomaly_status": label} for i, (_, label) in enumerate(image_data)])
    image_labels_df.to_csv(os.path.join(output_dir, f"{client_id}_image_labels.csv"), index=False)
    print(f"Synthetic data saved locally for {client_id} in {output_dir}/ (not committed to Git).")

if __name__ == "__main__":
    print("--- Generating Synthetic Data for Guardian AI Clients ---")
    client_ids_to_generate = ["client_A", "client_B", "client_C"]
    for client_id in client_ids_to_generate:
        print(f"\nGenerating data for {client_id}...")
        text_df = generate_synthetic_text_data(num_records=50, client_id=client_id, compliance_ratio=0.8)
        image_data = generate_synthetic_image_data(num_images=5, client_id=client_id)
        sensor_df = generate_synthetic_sensor_data(num_points=100, client_id=client_id)
        save_client_data_locally(client_id, text_df, image_data, sensor_df)

    print("\nVerification (Client A Text Data Head):")
    print(pd.read_csv("data/synthetic/client_A_text.csv").head())
    print("\nVerification (Client A Sensor Data Head):")
    print(pd.read_csv("data/synthetic/client_A_sensor.csv").head())