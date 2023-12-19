from app.analytics import cifar_cnn, mlflow_utils

mlflow_utils.start_new_root_run()
cifar_cnn.upload_dataset(\"{dataset-name}\")