import requests
import numpy as np
import tensorflow as tf
from pipeline.ocr.functions import align_image, process_output
from processing_tools import get_grpc_predict 

def corner(img, url_model, threshold, targetSize):
    """
    Detect corner from img

    Parameters:
        img: image tensor
        url_model: string url of model
        threshold: float, threshold of confidence
        targetSize: int, target size of image
    Returns:
        corner: float tensor of shape [1, 4]
    """
    image = np.expand_dims(img, axis=0)
    image = tf.make_tensor_proto(image)
    try:
        results = get_grpc_predict(url_model, 'input_tensor', image)
    except Exception as e: 
        print(e)

    results = results.outputs
    # payload = {'instances': [img.tolist()]}
    # res = requests.post(url_model, json=payload)
    # data= res.json()['predictions'][0]
    # results = process_output('corner', data, threshold, targetSize)

    results = process_output('corner', results, threshold, targetSize)
    
    crop_img = align_image(img, results)
    crop_img = np.array(crop_img)
    # return results
    return crop_img
    