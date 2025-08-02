# Guardian AI: Zero-Trust Multi-Modal Compliance & Risk Auditor

### Project Overview

This project, "Guardian AI," is a proof-of-concept MLOps pipeline designed to solve a critical real-world problem: using AI on sensitive, multi-modal data without compromising privacy, security, and compliance. It demonstrates a **Zero-Trust architecture**, ensuring that raw sensitive data never leaves its secure enclave.

The system processes multi-modal data (text, images, and sensor data) through a series of privacy-preserving layers, delivering high-utility insights without ever exposing the underlying sensitive details.

### Architecture

Our MLOps pipeline is built on a layered architecture that enforces privacy and security at every stage.



* **Layer 0: Secure Client Enclave** - Raw, sensitive data is generated and resides locally. It is never transmitted.
* **Layer 1: Privacy-Preserving Features** - Raw data is transformed into non-identifiable features. Key insights are encrypted using **Homomorphic Encryption (HE)** before being shared.
* **Layer 2: Federated Aggregation** - A central server orchestrates **Federated Learning (FL)**. Clients send encrypted model updates, which are aggregated without the server ever seeing raw data.
* **Layer 3: Auditable Insights** - The final, high-level insights are decrypted and presented in a user-friendly dashboard, with a full audit trail of the privacy-preserving process.

### Key MLOps Skills Demonstrated

* **Privacy-Preserving AI:** Implemented concepts of **Homomorphic Encryption (HE)** for encrypted computation and **Federated Learning (FL)** for decentralized model training.
* **Zero-Trust Architecture:** Designed a pipeline where no component is implicitly trusted, and data privacy is enforced by design.
* **Multi-Modal Data Handling:** Developed a system that can conceptually handle and process text, image, and sensor data.
* **CI/CD Automation:** Built a GitHub Actions workflow that automatically runs the FL simulation, logs results to **Weights & Biases (W&B)**, and deploys the application to **Hugging Face Spaces**.
* **Experiment Tracking:** Used **W&B** to monitor the performance of the federated model across training rounds on a public, non-sensitive dataset.

### How to Run the Project

* **Live Demo:** [https://huggingface.co/spaces/YOUR_GITHUB_USERNAME/guardian-ai-auditor-demo](https://huggingface.co/spaces/YOUR_GITHUB_USERNAME/guardian-ai-auditor-demo)
* **GitHub Repository:** [https://github.com/MukeshPyatla/guardian-ai-auditor](https://github.com/MukeshPyatla/guardian-ai-auditor)
* **W&B Project Dashboard:** [https://wandb.ai/YOUR_WANDB_USERNAME/guardian-ai-fl](https://wandb.ai/YOUR_WANDB_USERNAME/guardian-ai-fl)

**(Remember to replace the placeholder URLs with your actual information.)**