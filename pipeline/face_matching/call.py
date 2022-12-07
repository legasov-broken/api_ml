from multiprocessing import Process
from flask import jsonify, request, json
import numpy as np
from pipeline.face_matching.compare import findEuclideanDistance, img_to_encoding, l2_normalize, detect_face, distance_to_confident
from pipeline.face_matching.mtcnn.mtcnn import MTCNN
from pipeline.face_matching import app
import tensorflow as tf
tf.config.run_functions_eagerly(run_eagerly=True)

# pnet_url = 'http://localhost:8501/v1/models/p_net:predict'
# rnet_url = 'http://localhost:8501/v1/models/r_net/versions/1:predict'
# onet_url = 'http://localhost:8501/v1/models/o_net:predict'
# facenet_url = 'http://localhost:8501/v1/models/face_net:predict'


@app.route('/api/v1/face_matching', methods=['POST'])
def api_face_matching():
    """
    Compare two faces from two images

    Parameters:
        url_1: string url of image 1
        url_2: string url of image 2
        MTCNN model:
            pnet_url: string url of pnet model
            rnet_url: string url of rnet model
            onet_url: string url of onet model

    Returns:
        result: dictionary of result
    """
    data = json.loads(request.data)
    url_image1 = data['url_image1']
    url_image2 = data['url_image2']
    # Time to load images

    pnet_url = data['pnet_url']
    rnet_url = data['rnet_url']
    onet_url = data['onet_url']
    facenet_url = data['facenet_url']

    mtcnn = MTCNN(pnet_url, rnet_url, onet_url)
    try:
        face1 = detect_face(url_image1, mtcnn)
        face2 = detect_face(url_image2, mtcnn)
    except:
        return jsonify({'errorCode': 1, 'errorMessage': 'no face detected'})
    embedding_one = img_to_encoding(face1, facenet_url)
    embedding_two = img_to_encoding(face2, facenet_url)

    # Convert list to array
    embedding_one = np.array(embedding_one)
    embedding_two = np.array(embedding_two)
    # Calculate distance
    dist = findEuclideanDistance(l2_normalize(
        embedding_one), l2_normalize(embedding_two)).astype(np.float64)

    confident = distance_to_confident(dist, 1.15)

    return jsonify({'errorCode': 0, 'data': {'distance': dist, 'confident': confident}, 'errorMessage': 'success'})


if __name__ == '__main__':
    api_face_matching(
        'https://funflow-sp.sgp1.digitaloceanspaces.com/upload/blob_1f04290d-e3a9-4fbd-be2b-6cc3dee40e06',
        'https://funflow-sp.sgp1.digitaloceanspaces.com/upload/blob_1f04290d-e3a9-4fbd-be2b-6cc3dee40e06')
