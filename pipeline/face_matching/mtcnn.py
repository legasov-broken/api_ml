import time
import requests
import tensorflow as tf
from processing_tools import get_grpc_predict
from pipeline.face_matching.box_utils import calibrate_box, convert_to_square, get_image_boxes, generate_bboxes, preprocess
tf.config.run_functions_eagerly(run_eagerly=True)

DEF_THRESHOLDS = [0.6, 0.7, 0.7]
DEF_NMS_THRESHOLDS = [0.6, 0.6, 0.6]

class MTCNN(object):
    """ Top level class for mtcnn detection """
    def __init__(self, pnet_url, rnet_url, onet_url,
                 min_face_size=20.0,
                 thresholds=None,
                 nms_thresholds=None,
                 max_output_size=300):
        self.pnet_url = pnet_url
        self.rnet_url = rnet_url
        self.onet_url = onet_url
        self.min_face_size = min_face_size
        self.thresholds = thresholds or DEF_THRESHOLDS
        self.nms_thresholds = nms_thresholds or DEF_NMS_THRESHOLDS
        self.max_output_size = max_output_size
        self.scale_cache = {}

    def detect(self, img):
        """Detect faces and facial landmarks on an image

        Parameters:
            img: rgb image, numpy array of shape [h, w, 3]

        Returns:
            bboxes: float tensor of shape [n, 4], face bounding boxes
            landmarks: float tensor of shape [n, 10], 5 facial landmarks,
                        first 5 numbers of array are x coords, last are y coords
            scores: float tensor of shape [n], confidence scores
        """
        height, width, _ = img.shape
        img = tf.convert_to_tensor(img, tf.float32)
        scales = self.get_scales(height, width)
        start = time.time()
        bboxes = self.stage_one(img, scales)
        end = time.time()
        print('stage one: {}s'.format(end - start))
        if len(bboxes) == 0:
            return [], [], []
        bboxes = self.stage_two(img, bboxes, height, width, bboxes.shape[0])
        if len(bboxes) == 0:
            return [], [], []
        bboxes, landmarks, scores = self.stage_three(img, bboxes,
                                                     height, width, bboxes.shape[0])
        bboxes = bboxes[0]
        x = max(0, int(bboxes[0]))
        y = max(0, int(bboxes[1]))
        w = int(bboxes[2]) - x
        h = int(bboxes[3]) - y
        return (x,y,w,h)

    def get_scales(self, height, width):
        """Compute scaling factors for given image dimensions

        Parameters:
            height: float
            width: float

        Returns:
            list of floats, scaling factors
        """
        min_length = min(height, width)
        # typically scaling factors will not change in a video feed
        if min_length in self.scale_cache:
            return self.scale_cache[min_length]

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)
        # scales for scaling the image
        scales = []
        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / self.min_face_size
        min_length *= m
        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor**factor_count)
            min_length *= factor
            factor_count += 1

        self.scale_cache[min_length] = scales
        return scales

    # @tf.function(
    #     input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
    #                      tf.TensorSpec(shape=(), dtype=tf.float32),
    #                      tf.TensorSpec(shape=(), dtype=tf.float32),
    #                      tf.TensorSpec(shape=(), dtype=tf.float32)])
    def stage_one_scale(self, img, height, width, scale):
        """Perform stage one part with a given scaling factor

        Parameters:
            img: rgb image, float tensor of shape [h, w, 3]
            height: image height, float
            width: image width, float
            scale: scaling factor, float

        Returns:
            float tensor of shape [n, 9]
        """
        hs = tf.math.ceil(height * scale)
        ws = tf.math.ceil(width * scale)
        img_in = tf.image.resize(img, (hs, ws))
        img_in = preprocess(img_in)
        img_in = tf.expand_dims(img_in, 0)
        img_in = tf.make_tensor_proto(img_in)
        # img_in = tf.make_ndarray(img_in)

        # payload = {'instances': img_in.tolist()}
        # res = requests.post(self.pnet_url, json=payload)
        # res = res.json()['predictions'][0]
        result = get_grpc_predict(self.pnet_url, 'input_3', img_in)
        
        conv2d_12 = tf.make_ndarray(result.outputs['conv2d_12'])
        softmax_2 = tf.make_ndarray(result.outputs['softmax_2'])
        # print(softmax_2.shape)
        # print(conv2d_12.shape)
        # probs = softmax_2[0]
        # offsets = conv2d_12[0]

        probs = tf.convert_to_tensor(softmax_2[0])
        offsets = tf.convert_to_tensor(conv2d_12[0])
        # probs = tf.convert_to_tensor(res['softmax_2'])
        # offsets = tf.convert_to_tensor(res['conv2d_12'])

        boxes = generate_bboxes(probs, offsets, scale, self.thresholds[0])

        if len(boxes) == 0:
            return boxes
        keep = tf.image.non_max_suppression(boxes[:, :4], boxes[:, 4], self.max_output_size,
                                            iou_threshold=0.5)

        boxes = tf.gather(boxes, keep)
        return boxes

    # @tf.function(
    #     input_signature=[tf.TensorSpec(shape=(None, 9), dtype=tf.float32)])
    def stage_one_filter(self, boxes):
        """Filter out boxes in stage one 

        Parameters:
            boxes: collected boxes with different scales, float tensor of shape [n, 9]

        Returns:
            float tensor of shape [n, 4]
        """
        bboxes, scores, offsets = boxes[:, :4], boxes[:, 4], boxes[:, 5:]
        # use offsets predicted by pnet to transform bounding boxes
        bboxes = calibrate_box(bboxes, offsets)
        bboxes = convert_to_square(bboxes)

        keep = tf.image.non_max_suppression(bboxes, scores, self.max_output_size,
                                            iou_threshold=self.nms_thresholds[0])
        bboxes = tf.gather(bboxes, keep)
        return bboxes

    def stage_one(self, img, scales):
        """Run stage one on the input image

        Parameters:
            img: rgb image, float tensor of shape [h, w, 3]
            scales: scaling factors, list of floats

        Returns:
            float tensor of shape [n, 4], predicted bounding boxes
        """
        height, width, _ = img.shape
        boxes = []

        # run P-Net on different scales
        for s in scales:
            boxes.append(self.stage_one_scale(img, height, width, s))
        # collect boxes (and offsets, and scores) from different scales
        boxes = tf.concat(boxes, 0)
        if boxes.shape[0] == 0:
            return []
        return self.stage_one_filter(boxes)

    # @tf.function(
    #     input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
    #                      tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    #                      tf.TensorSpec(shape=(), dtype=tf.float32),
    #                      tf.TensorSpec(shape=(), dtype=tf.float32),
    #                      tf.TensorSpec(shape=(), dtype=tf.int32)])
    def stage_two(self, img, bboxes, height, width, num_boxes):
        """Run stage two on the input image

        Parameters:
            img: rgb image, float tensor of shape [h, w, 3]
            bboxes: bounding boxes from stage one, float tensor of shape [n, 4]
            height: image height, float
            width: image width, float
            num_boxes: number of rows in bboxes, int

        Returns:
            float tensor of shape [n, 4], predicted bounding boxes
        """
        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes, size=24)
        img_in = tf.make_tensor_proto(img_boxes)
        # img_in = tf.make_ndarray(img_in)

        # payload = {'instances': img_in.tolist()}
        # res = requests.post(self.rnet_url, json=payload)
        # res = res.json()['predictions']
        result = get_grpc_predict(self.rnet_url, 'input_1', img_in)

        dense5_2 = tf.make_ndarray(result.outputs['dense5_2'])
        softmax = tf.make_ndarray(result.outputs['softmax'])
        probs = tf.convert_to_tensor(softmax)
        offsets = tf.convert_to_tensor(dense5_2)
        # for i in range(len(res)):
        #     probs.append(res[i]['softmax'])
        #     offsets.append(res[i]['dense5_2'])

        # probs = tf.convert_to_tensor(probs)
        # offsets = tf.convert_to_tensor(offsets)

        keep = tf.where(probs[:, 1] > self.thresholds[1])[:, 0]

        bboxes = tf.gather(bboxes, keep)
        offsets = tf.gather(offsets, keep)
        scores = tf.gather(probs[:, 1], keep)

        bboxes = calibrate_box(bboxes, offsets)
        bboxes = convert_to_square(bboxes)

        keep = tf.image.non_max_suppression(bboxes, scores,
                                            self.max_output_size, self.nms_thresholds[1])
        bboxes = tf.gather(bboxes, keep)
        return bboxes

    # @tf.function(
    #     input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
    #                      tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    #                      tf.TensorSpec(shape=(), dtype=tf.float32),
    #                      tf.TensorSpec(shape=(), dtype=tf.float32),
    #                      tf.TensorSpec(shape=(), dtype=tf.int32)])
    def stage_three(self, img, bboxes, height, width, num_boxes):
        """Run stage three on the input image

        Parameters:
            img: rgb image, float tensor of shape [h, w, 3]
            bboxes: bounding boxes from stage two, float tensor of shape [n, 4]
            height: image height, float
            width: image width, float
            num_boxes: number of rows in bboxes, int

        Returns:
            bboxes: float tensor of shape [n, 4], face bounding boxes
            landmarks: float tensor of shape [n, 10], 5 facial landmarks,
                        first 5 numbers of array are x coords, last are y coords
            scores: float tensor of shape [n], confidence scores
        """
        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes, size=48)
        img_boxes = tf.make_tensor_proto(img_boxes)
        # img_boxes = tf.make_ndarray(img_boxes)
        # payload = {'instances': img_boxes.tolist()}
        # res = requests.post(self.onet_url, json=payload)
        # res = res.json()['predictions']
        result = get_grpc_predict(self.onet_url, 'input_2', img_boxes)

        dense6_2 = tf.make_ndarray(result.outputs['dense6_2'])
        softmax_1 = tf.make_ndarray(result.outputs['softmax_1'])
        dense6_3 = tf.make_ndarray(result.outputs['dense6_3'])
        probs = tf.convert_to_tensor(softmax_1)
        offsets = tf.convert_to_tensor(dense6_2)
        landmarks = tf.convert_to_tensor(dense6_3)
        # probs = []
        # offsets = []
        # landmarks = []
        # for i in range(len(res)):
        #     probs.append(res[i]['softmax_1'])
        #     offsets.append(res[i]['dense6_2'])
        #     landmarks.append(res[i]['dense6_3'])
        # probs = tf.convert_to_tensor(probs)
        # offsets = tf.convert_to_tensor(offsets)
        # landmarks = tf.convert_to_tensor(landmarks)
        keep = tf.where(probs[:, 1] > self.thresholds[2])[:, 0]
        bboxes = tf.gather(bboxes, keep)
        offsets = tf.gather(offsets, keep)
        scores = tf.gather(probs[:, 1], keep)
        landmarks = tf.gather(landmarks, keep)

        # compute landmark points
        width = tf.expand_dims(bboxes[:, 2] - bboxes[:, 0] + 1.0, 1)
        height = tf.expand_dims(bboxes[:, 3] - bboxes[:, 1] + 1.0, 1)
        xmin = tf.expand_dims(bboxes[:, 0], 1)
        ymin = tf.expand_dims(bboxes[:, 1], 1)
        landmarks = tf.concat([xmin + width * landmarks[:, 0:5],
                               ymin + height * landmarks[:, 5:10]], 1)

        bboxes = calibrate_box(bboxes, offsets)
        keep = tf.image.non_max_suppression(bboxes, scores,
                                            self.max_output_size, self.nms_thresholds[2])
        bboxes = tf.gather(bboxes, keep)
        landmarks = tf.gather(landmarks, keep)
        scores = tf.gather(scores, keep)
        return bboxes, landmarks, scores
