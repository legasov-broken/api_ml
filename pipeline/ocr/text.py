import numpy as np
import tensorflow as tf
import requests

from pipeline.ocr.functions import process_output
from processing_tools import get_grpc_predict

def text(img, url_model, threshold, targetSize):
    img = np.expand_dims(img, axis=0)
    img = tf.make_tensor_proto(img)
    
    results = get_grpc_predict(url_model, 'input_tensor', img)
    results = results.outputs
    results = process_output('text', results, threshold, targetSize)

    # payload = {'instances': [img.tolist()]}
    # res = requests.post(url_model, json=payload)
    # data= res.json()['predictions'][0]
    # results = process_output('text', results, threshold, targetSize)
    return results