import sys
sys.path.insert(0, './models')

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2


from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/botest2.pb'

# List of the strings that is used to add correct label for each box. frozen_inference_graph
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

print PATH_TO_LABELS

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_grap = fid.read()
        od_graph_def.ParseFromString(serialized_grap)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


PATH_TO_TEST_IMAGES_DIR = 'img'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'teste-{}.jpg'.format(i)) for i in range(1, 3) ]

print TEST_IMAGE_PATHS
IMAGE_SIZE = (12, 8)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8)

            print("image " + image_path.split('.')[0] + '_labeled.jpg')
            keywords = ""
            i = 0
            while (i < len(np.squeeze(scores))):
                currentScore = np.squeeze(scores)[i]
                if currentScore >= 0.75:
                    currentClasses = np.squeeze(classes).astype(np.int32)[i]
                    keywords += category_index[currentClasses]["name"] + ", "
                i = i + 1

            if len(keywords) >= 3:
                print(keywords[:-2])

            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            plt.show()
            


print 'Helloworld'

