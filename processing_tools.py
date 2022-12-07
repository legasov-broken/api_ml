import cv2
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np
import urllib


def url_to_image(url):
    """
    Read image from url
    """
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # image = cv2.imread(url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_parameters_from_model_url(url):
    """
    Get parameters from a model url.

    Parameters:
        url: str, url of the model

    Returns:
        dict, parameters of the model
    """
    params = {}

    netloc = url.split("//")[1].split("/")[0]
    params['netloc'] = netloc

    url = url.split("?")[1]
    url = url.split("&")
    for param in url:
        key, value = param.split("=")
        params[key] = value
    return params

def get_grpc_predict(url, input_name, input):
    """
    Get grpc predict stub
    
    Parameters:
        url: string, url of the server
        input_name: string, input name of the model
        input: numpy array, input data

    Returns:
        predict_stub: grpc stub
    """
    params = get_parameters_from_model_url(url)
    SERVER = params['netloc']
    GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3

    channel = grpc.insecure_channel(SERVER, options=[('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()

    request.model_spec.name = params['model_name']
    request.model_spec.signature_name = 'serving_default'
    if params['version'] is not None:
        request.model_spec.version.value = int(params['version'])

    request.inputs[input_name].CopyFrom(input)
    result = stub.Predict(request, 10.0)

    return result
