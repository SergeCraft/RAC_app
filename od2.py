import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# What model to download.
MODEL_NAME = 'rac_detection_200k'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/sal/projects/python/rac/rac_detection_200k/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/sal/projects/python/rac', 'label_map.pbtxt')

NUM_CLASSES = 1

PATH_TO_TEST_IMAGES_DIR = '/home/sal/Загрузки/testData'
#FILENAME = '1.jpg'

print(PATH_TO_TEST_IMAGES_DIR)
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Detection

if __name__ == "__main__":
	if len (sys.argv) > 1:
		FILENAME = sys.argv[1]
	else:
		FILENAME = '1.jpg'
	
	cap = cv2.VideoCapture(PATH_TO_TEST_IMAGES_DIR + '/' + FILENAME)
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			while True:
				ret, image_np = cap.read()
				
				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(image_np, axis=0)
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				  
				# Each box represents a part of the image where a particular object was detected.
				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
				 
				# Each score represent how level of confidence for each of the objects.
				# Score is shown on the result image, together with the class label.
				scores = detection_graph.get_tensor_by_name('detection_scores:0')
				classes = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			  
				# Actual detection.
				start = time.clock()
				(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
				print('detected for {} ms'.format(str(round((time.clock() - start)*1000, 1))))
	  
				# Visualization of the results of a detection.
				vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				category_index,
				use_normalized_coordinates=True,
				line_thickness=8)
				cv2.imshow('object detection', cv2.resize(image_np, (640,480)))
				if FILENAME[-3:] == 'jpg':
					cv2.waitKey(5000)
				if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break