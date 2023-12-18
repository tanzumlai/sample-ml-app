from app.analytics import cifar_cnn
import hickle as hkl
dataset_path=cifar_cnn.download_dataset("{dataset-name}")
data=hkl.load(dataset_path)
cifar_cnn.train_model("{model-name}", "{model-flavor}", "{model-stage}", data, epochs={epochs})