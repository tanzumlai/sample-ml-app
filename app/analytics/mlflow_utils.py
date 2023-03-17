import mlflow
import os
from mlflow import MlflowClient
from app.analytics import config


# ## Utilities
def get_run_for_artifacts(active_run_id):
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or 'Default'
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string="tags.mainartifacts='y'", max_results=1,
                              output_format='list')
    if len(runs):
        return runs[0].info.run_id
    else:
        mlflow.set_tags({'mainartifacts': 'y'})
        return active_run_id


def get_root_run(active_run_id=None):
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or 'Default'
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string="tags.runlevel='root'", max_results=1,
                              output_format='list')
    if len(runs):
        parent_run_id = runs[0].info.run_id
        mlflow.set_tags({'mlflow.parentRunId': parent_run_id})
        return parent_run_id
    else:
        mlflow.set_tags({'runlevel': 'root'})
        return active_run_id


def start_new_root_run():
    root_run_id = get_root_run()
    MlflowClient().set_terminated(get_root_run()) if root_run_id else True
    mlflow.set_tags({'runlevel': 'root'})


def get_current_run():
    last_active_run = mlflow.last_active_run()
    return last_active_run.info.run_id if last_active_run else None


def prep_mlflow_run(active_run):
    mlflow.set_tags({'mlflow.parentRunId': get_root_run(active_run_id=active_run.info.run_id)})


def get_experiment_metrics():
    model_uri = f"models:/{config.model_name}/{config.model_stage}"
    model_info = mlflow.models.get_model_info(model_uri)
    if model_info is None:
        return {}
    else:
        experiment_id = mlflow.get_run(model_info.run_id).info.experiment_id
        experiment_name = mlflow.get_experiment(f"{experiment_id}").name
        runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string="tags.runlevel='root'", max_results=1,
                                  output_format='list', order_by=["metrics.accuracy_score DESC"])
        return runs[0].data.metrics if len(runs) else {}
