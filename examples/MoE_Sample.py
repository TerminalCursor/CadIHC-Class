from histclass.interfaces.data.types import DataTypes
from histclass.backends.pytorch.models.utils import download_model
from histclass.backends.pytorch.data.dataset import ImageDataframe, PointcloudDataframe
from histclass.backends.pytorch.data.utils import get_data, get_metadata_labels
from histclass.backends.pytorch.models.vit import ViTModel
from histclass.tools.configuration import load_config
from glob import glob
from torch.nn.functional import softmax
import numpy as np

models_str = glob("ihc/models/classification/*2tn*")
models_str = [m.split("/")[-1] for m in models_str]
image_src = "ihc/data/classification/TMA00.B05/"

pred_data = []
for model_str in models_str:
    model = download_model(model_str)
    model.model.eval()
    patches = 1
    if isinstance(model, ViTModel):
        patches = model.image_size
    configuration = load_config(model.data_types, model.category_type, model.model_type, patches=patches)

    # Load image
    DataframeInit = (
        ImageDataframe
        if model.data_types in [DataTypes.ImageCNN, DataTypes.ImageViT]
        else PointcloudDataframe
    )
    metadata, labels = get_metadata_labels(image_src, model.category_type)
    assert metadata is not None
    data = get_data(image_src, model.data_types, metadata)
    assert data is not None
    dataframe = DataframeInit(
        image_src,
        model.data_types,
        model.category_type,
        name=image_src,
        data=data,
        labels=labels,
    )
    if model.data_types not in [DataTypes.ImageCNN, DataTypes.ImageViT]:
        assert isinstance(dataframe, PointcloudDataframe)
        assert configuration.pointcloud_config is not None
        dataframe.min_seq_length = configuration.pointcloud_config.min_points
        dataframe.seq_len = configuration.pointcloud_config.max_points
    else:
        assert isinstance(dataframe, ImageDataframe)
        assert configuration.image_values is not None
        dataframe.width = configuration.image_values.window_width
        dataframe.height = configuration.image_values.window_height

    frame_data = dataframe[0]
    input_data = frame_data[0]
    label = frame_data[1].detach().cpu().numpy().tolist()
    model = model.to(configuration.device)
    input_data = input_data.to(configuration.device)
    shape = input_data.shape
    input_data = input_data.reshape((1,*shape))
    prediction = model.forward(input_data)
    prediction = softmax(prediction, dim=-1)
    prediction = prediction.detach().cpu().numpy().flatten().tolist()
    pred_data.append((model_str, model.confidences, prediction, label))

model_names, model_confidences, predictions, labels = zip(*pred_data)
model_confidences = np.array(model_confidences)
predictions = np.array(predictions)

def bayes_3(
    model_names, model_confidences, predictions
):
    from numpy import exp, log, array

    models = model_names
    eps = 1e-12
    probs = predictions
    confidence = model_confidences
    raw_predictions = probs
    log_pred = log(raw_predictions + eps)
    weighted_log = confidence * log_pred
    log_post = weighted_log.sum(axis=0)
    post = exp(log_post - log_post.max())
    post = post / post.sum()

    prediction = post.argmax().item()
    top_models = weighted_log.argmax(axis=0)
    top_models = [models[int(_model)] for _model in top_models]
    metadata = {
        "top_model": top_models,
        "raw_prediction": post,
        "prediction": prediction,
    }
    return prediction, post, metadata

print(bayes_3(model_names, model_confidences, predictions), labels[0])
