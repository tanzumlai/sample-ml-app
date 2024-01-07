import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger().addHandler(logging.FileHandler(f"app.log"))

from app.analytics import preloader, cifar_cnn, config
import pickle
import os
import ray

# ## Initialize Ray environment
def initialize_environment(ray_address):
    logging.info(f"Initializing Ray environment - host={ray_address}...")
    ray.init(address=ray_address,
             runtime_env={'working_dir': ".", 'pip': "requirements.txt",
                          'env_vars': dict(os.environ),
                          'excludes': ['*.jar', '.git*/', 'jupyter/']}) if ray_address else ray.init() if not ray.is_initialized() else True

# ## Upload dataset

# Upload dataset to S3 via MlFlow
@ray.remote
def upload_dataset(dataset, dataset_url=None, ray_address=None):
    """
    Uploads the dataset.
    """
    return cifar_cnn.upload_dataset(dataset, dataset_url, to_parquet=True)


# ## Download DataSet
@ray.remote
def download_dataset(artifact, ray_address=None):
    """
    Downloads the dataset.
    """
    return cifar_cnn.download_dataset(artifact)


# ## Train Model
@ray.remote
def train_model(model_name, model_flavor, model_stage, data, epochs=10, ray_address=None):
    """
    Performs training on the provided CNN model.
    """
    return cifar_cnn.train_model(model_name, model_flavor, model_stage, data, epochs)


# ## Evaluate Model
@ray.remote
def evaluate_model(model_name, model_flavor, ray_address=None):
    """
    Evaluates the performance of the model based on specified criteria.
    """
    return cifar_cnn.evaluate_model(model_name, model_flavor)


# ## Promote Model to Staging
@ray.remote
def promote_model_to_staging(base_model_name, candidate_model_name, evaluation_dataset_name, model_flavor,
                             use_prior_version_as_base=False, ray_address=None):
    """
    Evaluates the performance of the currently trained candidate model compared to the base model.
    The model that performs better based on specific metrics is then promoted to Staging.
    """
    return cifar_cnn.promote_model_to_staging(base_model_name, candidate_model_name, evaluation_dataset_name,
                                              model_flavor,
                                              use_prior_version_as_base)


# ## Make Prediction
def predict(img, model_name, model_stage):
    """
    Returns a CIFAR 10 class label representing the model's classification of an image.
    """
    return cifar_cnn.predict(img, model_name, model_stage)