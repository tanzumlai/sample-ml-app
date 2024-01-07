from app.analytics import cifar_cnn, cifar_cnn_ray
import hickle as hkl
import sys

dataset_path = cifar_cnn.download_dataset(artifact=sys.argv[1])
data = hkl.load(dataset_path)

"""
" Launch a remote Ray task if a Ray address parameter was passed; 
" else, execute the task locally
"""
if len(sys.argv)>6 and sys.argv[6]:
    cifar_cnn_ray.initialize_environment(sys.argv[6])
    cifar_cnn_ray.train_model.remote(
        model_name=sys.argv[2],
        model_flavor=sys.argv[3],
        model_stage=sys.argv[4],
        data=data,
        epochs=int(sys.argv[5])
    )
else:
    cifar_cnn.train_model(
        model_name=sys.argv[2],
        model_flavor=sys.argv[3],
        model_stage=sys.argv[4],
        data=data,
        epochs=int(sys.argv[5])
    )
