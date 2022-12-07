import math
import cv2
import numpy as np
from processing_tools import get_grpc_predict, url_to_image
import tensorflow as tf

# from face_compare.model import facenet_model, img_to_encoding

# load model
# model1 = model.facenet_model(input_shape=(3, 96, 96))


def detect_face(url, model):
    """
    Detect face from url img

    Parameters:
        url: string url of image
        model: class MTCNN model for face detection

    Returns:
        face_image: face image
    """
    image = url_to_image(url)
    results = model.detect_faces(image)
    area_img = [result['box'][2] * result['box'][3] for result in results]

    max_area = max(area_img)
    max_index = area_img.index(max_area)
    if max_area < 3000:
        raise Exception('No face detected')
    x, y, w, h = results[max_index]['box']

    face = image[int(y):int(y)+int(h), int(x):int(x)+int(w)]
    # cv2.imwrite('face.jpg', face)
    return face


def img_to_encoding(img, facenet_url):
    """
    Encode image to 512-dimensional vector using facenet model

    Parameters:
        image: image tensor
        facenet_url: string url of facenet model

    Returns:
        encoding: float tensor of shape [1, 512]
    """
    target_size = (160, 160)
    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2),
                     (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

        # ------------------------------------------

        # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

        # ---------------------------------------------------

        # normalizing the image pixels

    # img_pixels = Image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    mean, std = img.mean(), img.std()
    img = (img - mean) / std

    # img_pixels /= 255 #normalize input in [0, 1]
    # payload = {'instances': img_pixels.tolist()}
    # res = requests.post(facenet_url, json=payload)
    # # embedding = model.predict(img_pixels)[0].tolist()
    # embedding = res.json()['predictions']
    # img_pixels = img_pixels.astype('float64')
    img = tf.make_tensor_proto(img, dtype=tf.float32)

    result = get_grpc_predict(facenet_url, 'input_1', img)

    embedding = tf.make_ndarray(result.outputs['Bottleneck_BatchNorm'])
    # resized = cv2.resize(image, (160, 160))
    # Swap channel dimensions
    # input_img = resized[...,::-1]
    # # Switch to channels first and round to specific precision.
    # input_img = np.around(np.transpose(input_img, (2,0,1))/255.0, decimals=12)
    # x_train = np.array([input_img])
    # embedding = model.predict_on_batch(x_train)
    return embedding


def findEuclideanDistance(source_embedding, test_embedding):
    """
    Find Euclidean distance between two embeddings

    Parameters:
        source_embedding: float tensor of shape [1, 512]
        test_embedding: float tensor of shape [1, 512]

    Returns:
        distance: float
    """
    if type(source_embedding) == list:
        source_embedding = np.array(source_embedding)

    if type(test_embedding) == list:
        test_embedding = np.array(test_embedding)

    euclidean_distance = source_embedding - test_embedding
    euclidean_distance = np.sum(np.multiply(
        euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def l2_normalize(x):
    """
    Normalize vector x.
    """
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def distance_to_confident(face_distance, threshold):
    """
    Convert distance to confidence using threshold.
    """
    if face_distance > threshold:
        range = (2 - threshold)
        linear_val = (2 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.5))
