import numpy as np
import cv2
import json
import urllib
import tensorflow as tf

def get_center_point(coordinate_dict):
    """
    convert (xmin, ymin, xmax, ymax) to (x_center, y_center)

    Parameters:
        coordinate_dict (dict): dictionary of coordinates

    Returns:
        points (dict): dictionary of coordinates
    """

    points = dict()
    for key in coordinate_dict.keys():
        xmin, ymin, xmax, ymax = coordinate_dict[key][0]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        points[key] = (x_center, y_center)
    return points

def find_miss_corner(coordinate_dict):
    """
    find the missed corner of a 
    
    Parameters:
        coordinate_dict (dict): dictionary of coordinates

    Returns:
        corner (str): name of the missed corner
    """

    dict_corner = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    for key in dict_corner:
        if key not in coordinate_dict.keys():
            return key

def calculate_missed_coord_corner(coordinate_dict):
    """
    calculate the missed coordinate of a corner

    Parameters:
        coordinate_dict (dict): dictionary of coordinates

    Returns:
        coordinate_dict (dict): dictionary of coordinates
    """
    #calulate a coord corner of a rectangle 
    def calculate_coord_by_mid_point(coor1, coord2, coord3):
        midpoint = np.add(coordinate_dict[coor1], coordinate_dict[coord2]) / 2
        y = 2 * midpoint[1] - coordinate_dict[coord3][1]
        x = 2 * midpoint[0] - coordinate_dict[coord3][0]
        return (x, y)
    # calculate missed corner coordinate
    corner = find_miss_corner(coordinate_dict)
    if corner == 'top_left':
        coordinate_dict['top_left'] = calculate_coord_by_mid_point('top_right', 
        'bottom_left', 'bottom_right')
    elif corner == 'top_right':
        coordinate_dict['top_right'] = calculate_coord_by_mid_point('top_left', 
        'bottom_right', 'bottom_left')
    elif corner == 'bottom_left':
        coordinate_dict['bottom_left'] = calculate_coord_by_mid_point('top_left', 
        'bottom_right', 'top_right')
    elif corner == 'bottom_right':
        coordinate_dict['bottom_right'] = calculate_coord_by_mid_point('bottom_left', 
        'top_right', 'top_left')
    return coordinate_dict

def perspective_transform(image, source_points):
    """
    perspective transform image

    Parameters:
        image (numpy array): base image
        source_points (numpy array): points of the image after detecting the corners

    Returns:
        image (numpy array): transformed image
    """

    # define the destination points (the points where the image will be mapped to)
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])

    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))
    return dst

def align_image(image, coordinate_dict):
    """
    align image (find the missed corner and perspective transform)

    Parameters:
        image (numpy array): image
        coordinate_dict (dict): dictionary of coordinates

    Returns:
        image (numpy array): image
    """

    if len(coordinate_dict) < 3:
        raise ValueError('Please try again')
    # convert (xmin, ymin, xmax, ymax) to (x_center, y_center)
    coordinate_dict = get_center_point(coordinate_dict)
    if len(coordinate_dict) == 3:
        coordinate_dict = calculate_missed_coord_corner(coordinate_dict)
    top_left_point = coordinate_dict['top_left']
    top_right_point = coordinate_dict['top_right']
    bottom_right_point = coordinate_dict['bottom_right']
    bottom_left_point = coordinate_dict['bottom_left']
    source_points = np.float32([top_left_point, top_right_point, bottom_right_point, bottom_left_point])
    # transform image and crop 
    crop = perspective_transform(image, source_points)
    return crop

def load_labels_map(label_url):
    # Opening JSON file
    f = open(label_url)
    data = json.load(f)
    if data:
        return data['labels']
    return []

def process_output(type, data, threshold, targetSize):
    """
    process output of model

    Parameters:
        type (str): type of model
        data (numpy array): output of model
        threshold (float): threshold of model
        targetSize (dict): target size of image

    Returns:
        result (dict): list bounding boxes
    """

    scores, boxes, classes = None, None, None
    label_map = None

    if type == 'corner':
        scores = list(tf.make_ndarray(data['detection_scores'])[0])
        boxes = list(tf.make_ndarray(data['detection_boxes'])[0])
        classes = list(tf.make_ndarray(data['detection_classes'])[0])
        # scores = np.array(data['detection_scores'])
        # boxes = np.array(data['detection_boxes'])
        # classes = np.array(data['detection_classes'])
        label_map = load_labels_map('pipeline/ocr/label_map/corner.pbtxt')
        # print(scores.shape, boxes.shape, classes.shape)
    if type == 'text':
        scores = list(tf.make_ndarray(data['detection_scores'])[0])
        boxes = list(tf.make_ndarray(data['detection_boxes'])[0])
        classes = list(tf.make_ndarray(data['detection_classes'])[0])
        label_map = load_labels_map('pipeline/ocr/label_map/text.pbtxt')

    results = {}
    for i in range(len(scores)):
        if scores[i] > threshold:
            label = label_map[int(classes[i]) - 1]
            if label in results:
                results[label].append([
                    boxes[i][1] * targetSize['w'], 
                    boxes[i][0] * targetSize['h'], 
                    boxes[i][3] * targetSize['w'], 
                    boxes[i][2] * targetSize['h']])
            else:
                results[label] = [[
                    boxes[i][1] * targetSize['w'], 
                    boxes[i][0] * targetSize['h'], 
                    boxes[i][3] * targetSize['w'], 
                    boxes[i][2] * targetSize['h']]]
    
    return results