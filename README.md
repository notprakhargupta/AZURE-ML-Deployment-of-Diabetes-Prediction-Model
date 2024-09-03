# AZURE-ML-Deployment-of-Diabetes-Prediction-Model
This project demonstrates the deployment of a diabetes prediction machine learning model using Azure.
# Diabetes Prediction Model Deployment

This repository demonstrates the process of deploying a diabetes prediction machine learning model using Azure Machine Learning. The project includes a complete workflow from model preparation to deployment and testing as a web service.

## Project Overview

The goal of this project is to deploy a trained machine learning model for predicting diabetes as a RESTful web service using Azure Machine Learning (Azure ML). The deployment is performed on Azure Container Instances (ACI), providing a scalable and cost-effective way to serve predictions.

## Project Structure

- `model.pkl`: The trained machine learning model saved in pickle format.
- `score.py`: Scoring script for loading the model and handling incoming prediction requests.
- `environment.yml`: Conda environment file specifying the dependencies required for the model.
- `deploy.py`: Python script for deploying the model to Azure Container Instances.
- `test_service.py`: Script for testing the deployed web service.

## Prerequisites

1. **Azure Machine Learning Workspace**: You need an Azure ML workspace. Create one via the Azure portal or use an existing workspace.
2. **Azure Subscription**: An Azure subscription to create resources like Azure Container Instances.
3. **Python**: Python 3.x with required libraries installed.

## Setup and Configuration

1. **Install Required Libraries**:

    Ensure you have the Azure Machine Learning SDK installed. You can install it using pip:
    ```bash
    pip install azureml-sdk
    ```

2. **Register the Model**:

    Use the following Python script to register your model with Azure ML:
    ```python
    from azureml.core import Workspace
    from azureml.core.model import Model

    # Connect to your Azure ML workspace
    ws = Workspace.from_config()

    # Register the model
    model = Model.register(workspace=ws,
                           model_path="model.pkl",
                           model_name="diabetes_prediction_model")
    ```

3. **Prepare the Scoring Script**:

    Create a file named `score.py` with the following content:
    ```python
    import json
    import joblib
    import numpy as np
    from azureml.core.model import Model

    def init():
        global model
        model_path = Model.get_model_path('diabetes_prediction_model')
        model = joblib.load(model_path)

    def run(raw_data):
        try:
            data = json.loads(raw_data)['data']
            data = np.array(data).reshape(1, -1)
            result = model.predict(data)
            return json.dumps({"result": result.tolist()})
        except Exception as e:
            return json.dumps({"error": str(e)})
    ```

4. **Create the Environment**:

    Define your environment in `environment.yml`:
    ```yaml
    name: myenv
    dependencies:
      - python=3.8
      - scikit-learn
      - numpy
      - pip:
        - azureml-defaults
    ```

5. **Deploy the Model**:

    Use the following script (`deploy.py`) to deploy the model to Azure Container Instances:
    ```python
    from azureml.core import Workspace
    from azureml.core.model import Model
    from azureml.core.environment import Environment
    from azureml.core.conda_dependencies import CondaDependencies
    from azureml.core.webservice import AciWebservice, Webservice
    from azureml.core.model import InferenceConfig

    # Connect to the workspace
    ws = Workspace.from_config()

    # Define environment
    env = Environment.from_conda_specification(name="myenv", file_path="environment.yml")

    # Define inference configuration
    inference_config = InferenceConfig(entry_script="score.py", environment=env)

    # Define ACI configuration
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, auth_enabled=True)

    # Deploy the model
    service = Model.deploy(workspace=ws,
                           name="diabetes-prediction-service",
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=aci_config)

    service.wait_for_deployment(show_output=True)
    print(service.state)
    ```

6. **Test the Service**:

    Use `test_service.py` to send a test request to the deployed web service:
    ```python
    import requests
    import json

    # Define the input data
    input_data = [5, 166, 72, 19, 175, 25.8, 0.587, 51]

    # Convert the input data to JSON
    input_data_json = json.dumps({"data": [input_data]})

    # Define the scoring URI
    scoring_uri = "<your-scoring-uri>"

    # Define the headers
    headers = {"Content-Type": "application/json"}

    # Send the request
    response = requests.post(scoring_uri, data=input_data_json, headers=headers)

    # Check the response
    if response.status_code == 200:
        result = response.json()
        print(result)
        prediction = result["result"][0]
        print(f"Prediction: {prediction}")
    else:
        print(f"Error: {response.text}")
    ```

## Deployment and Testing

1. **Deploy the Model**: Run `deploy.py` to deploy the model to Azure Container Instances.
2. **Test the Service**: Run `test_service.py` to verify the deployed web service is working as expected.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Azure Machine Learning for providing cloud-based model deployment services.
- Scikit-learn for the machine learning model.

