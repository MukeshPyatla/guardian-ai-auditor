from phe import paillier

public_key_global, private_key_global = None, None

def generate_global_paillier_keys():
    """Generates global Paillier public and private keys."""
    global public_key_global, private_key_global
    if public_key_global is None or private_key_global is None:
        public_key_global, private_key_global = paillier.generate_paillier_keypair()
    return public_key_global, private_key_global

def encrypt_value(value, public_key):
    """Encrypts a single numerical value."""
    return public_key.encrypt(value)

def decrypt_value(encrypted_value, private_key):
    """Decrypts a single encrypted numerical value."""
    return private_key.decrypt(encrypted_value)

def homomorphic_add_values(encrypted_val1, encrypted_val2):
    """Performs homomorphic addition on two encrypted values."""
    return encrypted_val1 + encrypted_val2

def homomorphic_multiply_by_scalar(encrypted_val, scalar):
    """Performs homomorphic multiplication by a scalar."""
    return encrypted_val * scalar

if __name__ == "__main__":
    pub_key, priv_key = generate_global_paillier_keys()
    print("Paillier Keys Generated (simulated global keys for demo).")
    value1 = 10.5
    value2 = 20.3
    scalar = 2.0
    enc_value1 = encrypt_value(value1, pub_key)
    enc_value2 = encrypt_value(value2, pub_key)
    print(f"\nOriginal value 1: {value1}")
    print(f"Original value 2: {value2}")
    enc_sum = homomorphic_add_values(enc_value1, enc_value2)
    dec_sum = decrypt_value(enc_sum, priv_key)
    print(f"Homomorphic Sum: Encrypted -> Decrypted = {dec_sum} (Expected: {value1 + value2})")
    enc_product = homomorphic_multiply_by_scalar(enc_value1, scalar)
    dec_product = decrypt_value(enc_product, priv_key)
    print(f"Homomorphic Product: Encrypted -> Decrypted = {dec_product} (Expected: {value1 * scalar})")
    print("\n--- Project Context ---")
    print("In this project, we'll conceptualize using HE for aggregating simple numerical insights or model parameters.")
    print("Full HE for complex AI models (like deep learning) is computationally intensive and not feasible for free tiers.")
    print("We demonstrate the *principle* of HE here to show privacy-preserving computation.")