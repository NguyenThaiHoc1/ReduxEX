import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

sys.path.insert(0, './models')

import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
 
from object_detection.utils import label_map_util
 
from object_detection.utils import visualization_utils as vis_util

# function need to project 
def save_Template_list_to_array(array_element, index):
    test = []
    for number in array_element[:, :, index]:
        for number_element in number:
            test.append(number_element)
    return np.array(test)

def reshape_numpyarray(numpyarrayItem, shapesize):
    return numpyarrayItem.reshape(shapesize)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size;
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_image_into_numpy_opencv(img):
    height, width = img.shape[:2]
    return np.array(img).reshape((height, width, 3)).astype(np.uint8)
def grayscale_image_load_data(path, gratmode=False, target_size=None):
    # muc tieu cua ham nay la viec minh lam mo tam anh xuong den muc toi thieu
    # dung dinh dang khuon mat de lay nhung box 
    # sau do lam mo den het muc co the 
    # tu do ta se dua vao data traning de co the biet cam xuc cua
    # moi box do la gi ?
    # muc tieu la lam sao co the nhan dien het tat ca khuon mat do
    # do cung la 1 ky thuat doi hoi su chinh xac
    pilgray_image = image.load_img(path, gratmode, target_size)
    return image.img_to_array(pilgray_image)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# function process label text for image
def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
    elif dataset_name == 'imdb':
        return {0:'woman', 1:'man'}
    elif dataset_name == 'KDEF':
        return {0:'AN', 1:'DI', 2:'AF', 3:'HA', 4:'SA', 5:'SU', 6:'NE'}
    else:
        raise Exception('Invalid dataset name')
# -------------------------
# loading detection emotion
emotion_model_path = './ssd_mobilenet_v1_coco_11_06_2017/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]
print emotion_target_size
# ---------------------------
PATH_TO_CKPT = "ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph_face.pb"
PATH_TO_LABELS = os.path.join('data', 'face_map_label.pbtxt')
NUM_CLASSES = 2
 
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
#load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
 
 
font = ImageFont.truetype('Library/Fonts/BigCaslon.ttf', 40)
font1 = cv2.FONT_HERSHEY_SIMPLEX



cap = cv2.VideoCapture('./img/2.mp4')
i = 0;
while(cap.isOpened()):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            ret, imagesx = cap.read()
            i += 1
            numpyarrayList = save_Template_list_to_array(imagesx, 0); # 0 hoac 1 tuy theo muon truyen gi vao cung dc
            ItemforMatrix = reshape_numpyarray(numpyarrayList, imagesx.shape[:2])
            imagesx[:, :, 0] = imagesx[:, :, 2]
            imagesx[:, :, 2] = ItemforMatrix
            imagesx = imagesx.astype(np.uint8)
            image_np_expanded = np.expand_dims(imagesx, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
    image_np_array1 = imagesx
    image_np_array1 = cv2.cvtColor(image_np_array1, cv2.COLOR_BGR2GRAY)
    print np.squeeze(scores) 
    height, width = imagesx.shape[:2]
    for img_number in range(boxes.shape[1]):
        if np.squeeze(scores)[img_number] <= 0.09:
            continue
        ymin, xmin, ymax, xmax = boxes[0, img_number, :]
        (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
        left = left.astype(np.int32)
        right = right.astype(np.int32)
        top = top.astype(np.int32)
        bottom = bottom.astype(np.int32)
        gray_face = image_np_array1[top:bottom, left:right]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]
        cv2.rectangle(imagesx, (left, top), (right, bottom), (255, 255, 0), 3)
        cv2.putText(imagesx, emotion_text,(left,bottom), font1, 1,(0,255,0),2, 1)
    #cv2.imshow('frame',imagesx)
    cv2.imwrite('./MTP/rslx-{}.jpg'.format(i),imagesx)
    key = cv2.waitKey(1) & 0xFF
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
print("\n\nBye bye\n")

