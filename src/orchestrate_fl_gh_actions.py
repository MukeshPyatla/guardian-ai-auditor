import subprocess
import sys
import os
import time
import threading
import pandas as pd
import numpy as np

# Add parent directory to path to import common and client_logic/server_logic modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from client_logic.data_generator import generate_synthetic_text_data, generate_synthetic_image_data, generate_synthetic_sensor_data, save_client_data_locally
from client_logic.he_utils import generate_global_paillier_keys

def run_fl_simulation(num_rounds=3, num_clients=3):
    print("--- Starting Federated Learning Simulation for GitHub Actions ---")

    # 1. Generate synthetic data for all clients for this run
    client_ids = [f"client_{chr(65 + i)}" for i in range(num_clients)]
    print(f"Generating synthetic data for clients: {client_ids}...")
    for client_id in client_ids:
        text_df = generate_synthetic_text_data(50, client_id, compliance_ratio=0.8)
        image_data = generate_synthetic_image_data(5, client_id)
        sensor_df = generate_synthetic_sensor_data(100, client_id)
        save_client_data_locally(client_id, text_df, image_data, sensor_df)
    print("Synthetic client data generated.")

    # 2. Start FL Server in a separate thread
    server_process = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), 'server_logic', 'fl_server.py')],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    server_thread = threading.Thread(target=read_process_output, args=(server_process, "SERVER"))
    server_thread.start()
    print("FL Server started in background.")
    time.sleep(10)  # Give server time to initialize

    # 3. Start FL Clients in separate processes
    client_processes = []
    for client_id in client_ids:
        client_process = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(__file__), 'client_logic', 'fl_client.py'), client_id],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        client_processes.append(client_process)
        client_thread = threading.Thread(target=read_process_output, args=(client_process, f"CLIENT_{client_id}"))
        client_thread.start()
        print(f"Client {client_id} started in background.")
        time.sleep(5)  # Stagger client starts

    # 4. Wait for all clients to finish
    print("\nWaiting for FL clients to complete their rounds...")
    for p in client_processes:
        try:
            p.wait(timeout=300)  # Wait up to 5 minutes per client
        except subprocess.TimeoutExpired:
            print(f"Client process timed out: {p.args}")
            p.kill()  # Terminate if it's stuck

    # 5. Wait for server to finish
    print("\nWaiting for FL Server to complete...")
    try:
        server_process.wait(timeout=300) # Wait up to 5 minutes for server
    except subprocess.TimeoutExpired:
        print("Server process timed out.")
        server_process.kill()

    print("\n--- Federated Learning Simulation Complete for GitHub Actions ---")

def read_process_output(process, name):
    """Reads and prints output from a subprocess in a non-blocking way."""
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(f"[{name}] {line}")
        sys.stdout.flush()
    process.stdout.close()

if __name__ == "__main__":
    # Ensure global Paillier keys are generated before FL starts
    generate_global_paillier_keys()
    run_fl_simulation(num_rounds=3, num_clients=3)