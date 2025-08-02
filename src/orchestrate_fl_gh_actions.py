import subprocess
import sys
import os
import time
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from client_logic.he_utils import generate_global_paillier_keys

def run_fl_simulation(num_rounds=3, num_clients=3):
    print("--- Starting Federated Learning Simulation for GitHub Actions ---")

    client_ids = [f"client_{chr(65 + i)}" for i in range(num_clients)]
    print(f"Clients for this run: {client_ids}")

    # 1. Start the FL Server in a background process
    server_process = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), 'server_logic', 'fl_server.py')],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    print("FL Server started in background. Waiting for 20 seconds for it to become ready...")
    time.sleep(20) # Add a significant delay to ensure the server is up

    # 2. Start FL Clients in separate processes
    client_processes = []
    for client_id in client_ids:
        client_process = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(__file__), 'client_logic', 'fl_client.py'), client_id],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        client_processes.append(client_process)
        print(f"Client {client_id} started in background.")
        time.sleep(5)  # Stagger client starts

    # 3. Wait for all clients to finish
    print("\nWaiting for FL clients to complete their rounds...")
    for p in client_processes:
        try:
            p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            print(f"Client process timed out: {p.args}")
            p.kill()

    # 4. Wait for server to finish
    print("\nWaiting for FL Server to complete...")
    try:
        server_process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        print("Server process timed out.")
        server_process.kill()

    print("\n--- Federated Learning Simulation Complete for GitHub Actions ---")

if __name__ == "__main__":
    generate_global_paillier_keys()
    run_fl_simulation(num_rounds=3, num_clients=3)
