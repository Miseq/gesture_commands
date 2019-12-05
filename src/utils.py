"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
import tensorflow as tf
from src.config import HAND_GESTURES
import keyboard
import time


def is_in_square(points, square):
    hand_x, hand_y = points
    (xa, ya), (xb, yb), (xc, yc), (xd, yd) = square
    max_x = max(xa,xb,xc,xd)
    min_x = min(xa, xb,xc,xd)
    max_y = max(ya,yb,yc,yd)
    min_y = min(ya,yb,yc,yd)
    if(hand_x >= min_x and hand_x<= max_x and hand_y >= min_y and hand_y <= max_y):
        return True
    else:
        return False

def load_graph(path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    return detection_graph, sess


def detect_hands(image, graph, sess):
    input_image = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    image = image[None, :, :, :]
    boxes, scores, classes = sess.run([detection_boxes, detection_scores, detection_classes],
                                      feed_dict={input_image: image})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


def predict(boxes, scores, classes, threshold, width, height, num_hands=2):
    count = 0
    results = {}
    for box, score, class_ in zip(boxes[:num_hands], scores[:num_hands], classes[:num_hands]):
        if score > threshold:
            y_min = int(box[0] * height)
            x_min = int(box[1] * width)
            y_max = int(box[2] * height)
            x_max = int(box[3] * width)
            category = HAND_GESTURES[int(class_) - 1]
            results[count] = [x_min, x_max, y_min, y_max, category]
            count += 1
    return results

def mimic(v, lock):

    while not False:
        if v.value!=0:
            combination=''
            if v.value==1: combination=('space')
            elif v.value==2: combination=('alt+tab')
            elif v.value==3: combination=("alt+shift+tab")
            elif v.value==4: combination=('win+d')

            keyboard.press(combination)
            keyboard.release(combination)
            with lock:
                v.value = 0
            time.sleep(1.5)