from model import train_model,trans_model
from configs import ModelConfigs
from mltu.tensorflow.callbacks import Model2onnx
configs = ModelConfigs()

model = trans_model(
    input_dim = (configs.height, configs.width, 3),
    output_dim = 78,
)
model.load_weights("Models\\03_handwriting_recognition\\202310201149\model.h5")
Model2onnx.model2onnx(model,'Models\\03_handwriting_recognition\\202310201059\model.onnx')