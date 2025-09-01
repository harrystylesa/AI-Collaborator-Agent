# AI-Collaborator Backend

A Python-based backend developed with **FastAPI** that serves as an API gateway for the [Google Docs Clone project](https://github.com/harrystylesa/google_docs_clone). This service orchestrates AI-driven summarization, model management, and real-time performance monitoring. 

🔗 Live Demo: [google-docs-clone-link](https://google-docs-clone-kappa-lovat.vercel.app) 

🔗 Frontend Repo: [Google Docs Clone](https://github.com/harrystylesa/google_docs_clone)

## 🏗️ Project Architecture
The architecture is designed to handle requests securely and efficiently, routing user actions from the frontend to the appropriate AI services.
![flow](https://i.ibb.co/ZRLp4wff/Untitled-diagram-Mermaid-Chart-2025-08-29-185644.png)


## 🛠 Tech Stack

**Backend & AI**
* **Python & FastAPI** – For building a high-performance API.
* **Databricks Model Serving** – To serve production-ready AI models with high reliability and scalability.
* **OpenAI API** –  For generating document summaries using powerful language models.
* **MLflow** –  For comprehensive model lifecycle management, including experiment tracking and model registry.

**DevOps & MLOps**
* **Docker & Kubernetes** –  For containerization and orchestration, ensuring a consistent and scalable deployment environment.
* **CI/CD** – Automated pipelines with GitHub Actions for fast and reliable deployments.

## ✨ Features & MLOps Practices

* **Model Serving**: The FastAPI backend acts as a proxy, intelligently routing summarization requests to the Databricks Model Serving Endpoint. This architecture decouples the AI model from the application logic, allowing for independent scaling and updates.
* **A/B Testing**: Implemented a data-driven feedback loop with A/B testing to continuously evaluate the performance of different model versions. This approach ensures that performance improvements are verified before a full rollout to all users.
* **Performance Monitoring**: Key metrics such are monitored in real-time using Databricks Inference Tables. This is crucial for detecting performance regressions or scaling issues early.
![memory usage](https://i.ibb.co/Y7ggksSg/2025-08-31-18-57-36.png)
![cpu usage](https://i.ibb.co/k2mnYw2n/2025-08-31-19-05-51.png)

* **Automated Deployment**: The backend service is deployed automatically to a Kubernetes cluster via GitHub Actions upon any code changes, enabling a fast and reliable CI/CD workflow.


## 🔐 Security & Deployment
The application is deployed on a Google Kubernetes Engine (GKE) cluster. To ensure robust security and availability, leverage Cloudflare as a proxy for the following practices:

* **SSL Termination**: Cloudflare handles SSL termination, providing HTTPS security and offloading encryption processing from the GKE cluster.

* **IP Protection**: It acts as a shield, preventing direct exposure of the GKE cluster's public IP and protecting the application from common web attacks.

## Project Structure

```
fastapi-project
├── databricks
│   └── models          # Databricks pyfunc model wrapper
│       └── summarization_agent
│           ├── log_model_with_prompt.py
│           ├── v1_prompts.txt
│           └── v2_prompts.txt
├── deploy.yaml # GKE deploy
├── Dockerfile
├── experiment.config.json   # A/B testing
├── kustomization.yaml
├── main.py            # Entry point of the FastAPI 
├── README.md          # Project documentation
├── requirements.txt   # Project dependencies
└── tools.py
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fastapi-project
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Submitting custome models to databricks
For submit models to databricks, create .env.model file in application folder, and fill following variables.

```shell
MLFLOW_TRACKING_URI=
DATABRICKS_HOST=
DATABRICKS_TOKEN=
MLFLOW_EXPERIMENT_ID=
OPENAI_API_KEY=
```

Run following command in your terminal to submit model.

```shell
python databricks/models/log_model_with_prompt.py
```
### Start FastAPI application
Create a .env.fastapi file in application folder, and fill in following variables.

```shell
DATABRICKS_TOKEN=
DATABRICKS_WAREHOUSE_HTTP_PATH=
DATABRICKS_HOST=
CLERK_SECRET_KEY=
FRONTEND_ORIGIN=
FRONTEND_ORIGIN_DEV=
OPENAI_API_KEY=
```

To run the FastAPI application, execute the following command:
```
uvicorn app.main:app --reload
```

You can access the API documentation at `http://127.0.0.1:80/docs`.