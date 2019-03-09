#imports

import os
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QFileDialog
import sys
import cv2
import numpy as np
import threading
import time
import queue
import collections
import json

#global variables

path_to_frozen_graph = 'rac_detection_200k/frozen_inference_graph.pb'
path_to_label_map = 'label_map.pbtxt'
classes_qty = 1
label_map = label_map_util.load_labelmap(path_to_label_map)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=classes_qty, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

ui = uic.loadUiType("RAC_app_v03.ui")[0]
console_queue = queue.Queue()
display_queue = queue.Queue()
player_list = []
source = None
cap_det_daemon = None
mode = 'Video file'
prev_row_length = 0
player_queue = queue.Queue()
last_update_frame_time = time.clock()
picture_list = []
current_picture_id_to_display = 0
capture = cv2.VideoCapture('')
playing = True
min_score = 0.2
RAC_qty = 0
RAC_detected = False		
last_RAC_frames_back = 0

#classes

class RAC_app_window(QMainWindow, ui):
	def __init__(self, parent=None):
		global cap_det_daemon, source
		QMainWindow.__init__(self, parent)
		self.setupUi(self)
		cap_det_daemon = threading.Thread(target = capture_and_detection_daemon, daemon = True)
		cap_det_daemon.start()
		
		self.window_width = self.video.frameSize().width()
		self.window_height = self.video.frameSize().height()
		self.fps_spinBox.setValue(25)
		self.fps_spinBox.setMaximum(25)
		self.fps_spinBox.setMinimum(1)
		self.video = OwnImageWidget(self.video)
		self.actionVideo_file.triggered.connect(self.select_mode)
		self.actionPictures.triggered.connect(self.select_mode)
		self.actionWeb_Camera.triggered.connect(self.select_mode)
		self.prev_button.clicked.connect(self.picture_control)
		self.next_button.clicked.connect(self.picture_control)
		self.play_button.clicked.connect(self.video_control)
		self.prev_button.hide()
		self.next_button.hide()
		self.play_button.hide()
		self.fps_spinBox.hide()
		self.label.hide()
        	
		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.player)
		self.timer.timeout.connect(self.update_frame)
		self.timer.timeout.connect(self.update_console)
		self.timer.start(1)
		c('Form loaded...')
			

	def player(self):
		global player_list, display_queue, mode, player_queue, last_update_frame_time, current_picture_id_to_display, picture_list, playing, last_RAC_frames_back, RAC_detected, RAC_qty
		if mode == 'Video file':
			if time.clock() - last_update_frame_time >= 1/float(self.fps_spinBox.value()):
				if not player_queue.empty() and playing:
					frame = player_queue.get()
					img = frame["img"]
					score = frame["score"]
					img_height, img_width, img_colors = img.shape
					scale_w = float(self.window_width) / float(img_width)
					scale_h = float(self.window_height) / float(img_height)
					scale = min([scale_w, scale_h])					
					if last_RAC_frames_back == 0 and score > min_score:
						last_RAC_frames_back = 1
					elif 0 < last_RAC_frames_back <= 10 and score > min_score:
						#c('RAC detected!')
						RAC_detected = True
						last_RAC_frames_back = 1									
					elif 0 < last_RAC_frames_back <= 10 and score <= min_score:
						#c('Waiting next RAC detection...', True)
						last_RAC_frames_back += 1
					elif last_RAC_frames_back > 10:
						last_RAC_frames_back = 0
						if RAC_detected:
							RAC_qty +=1
							self.label.setText('Сцепок: {0}'.format(RAC_qty))
							c('RAC quantity: {0}'.format(RAC_qty), True)
							RAC_detected = False
					if scale == 0:
						scale = 1			
					try:
						img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
						img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
						height, width, bpc = img.shape
						bpl = bpc * width						
						image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
						display_queue.put(image)						
						self.play_button.setText('Pause')
					except AttributeError as e:
						c('AttributeError occured')
					except BaseException as e:
						c(str(e))				
				else:
					self.play_button.setText('Play')
				
		elif mode == 'Pictures':
			active_queue = player_queue
			while not active_queue.empty():
				picture_list.append(active_queue.get())
			if picture_list:
				frame = picture_list[current_picture_id_to_display]
				img = frame["img"]
				img_height, img_width, img_colors = img.shape
				scale_w = float(self.window_width) / float(img_width)
				scale_h = float(self.window_height) / float(img_height)
				scale = min([scale_w, scale_h])
				if scale == 0:
					scale = 1			
				try:
					img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
					img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					height, width, bpc = img.shape
					bpl = bpc * width						
					image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
					display_queue.put(image)
				except AttributeError as e:
					c('AttributeError occured')
				except BaseException as e:
					c(str(e))
					
		elif mode == 'Web Camera':
			active_queue = player_queue
			if not active_queue.empty():
				frame = active_queue.get()
				img = frame["img"]
				img_height, img_width, img_colors = img.shape
				scale_w = float(self.window_width) / float(img_width)
				scale_h = float(self.window_height) / float(img_height)
				scale = min([scale_w, scale_h])
				if scale == 0:
					scale = 1			
				try:
					img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
					img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					height, width, bpc = img.shape
					bpl = bpc * width						
					image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
					display_queue.put(image)
				except AttributeError as e:
					c('AttributeError occured')
				except BaseException as e:
					c(str(e))
			
	def picture_control(self):
		global current_picture_id_to_display, picture_list
		sender = self.sender()
		if sender.text() == 'Prev':
			current_picture_id_to_display -= 1
		elif sender.text() == 'Next':
			current_picture_id_to_display += 1
		if abs(current_picture_id_to_display) >= len(picture_list):
			current_picture_id_to_display = 0
			
	def video_control(self):
		global playing
		if playing == True:
			playing = False
			self.play_button.setText('Play')
		else:
			playing = True
			self.play_button.setText('Pause')
		
		

	def update_frame(self):
		global display_queue, last_update_frame_time, playing
		if not display_queue.empty():
			#c('Display queue is not empty!', True)
			self.video.setImage(display_queue.get())
			last_update_frame_time = time.clock()
		
			
	
	def update_console (self):
		global prev_row_length
		#
		#if replace_last and prev_row_length == len(text):
			#w.console.setText('111')
		#
		if not console_queue.empty():
			text, replace_last = console_queue.get()
			current_row_length = len(text)
			if replace_last and current_row_length == prev_row_length:				
				self.console.setText(self.console.toPlainText()[:-(prev_row_length + 1)])
				prev_row_length = current_row_length
				self.console.append(text)
				self.console.moveCursor(11)	
			else:
				prev_row_length = current_row_length	
				self.console.append(text)
			
	def open_file_dialog(self):
		global mode
		if mode == 'Video file':
			file_list = QFileDialog.getOpenFileName(self, 'Open file', '/home/sal/Загрузки/testData')[0]
			c('Opened file {}'.format(file_list))
		elif mode == 'Pictures':			
			file_list = QFileDialog.getOpenFileNames(self, 'Open file', '/home/sal/Загрузки/testData')[0]
			for file_name in file_list:
				c('Opened file {}'.format(file_name))
		return file_list
			
	def select_mode(self):
		global mode, source, display_queue, current_picture_id_to_display, picture_list, capture, player_queue, RAC_qty
		sender = self.sender()
		mode = sender.text()
		c('Mode selected: {0}'.format(mode))
		player_queue = queue.Queue()
		display_queue = queue.Queue()
		self.video.image = None
		self.video.update()
		picture_list.clear()
		current_picture_id_to_display = 0
		capture.release()
		RAC_qty = 0
		if mode == 'Video file':
			source = self.open_file_dialog()
			self.prev_button.hide()
			self.next_button.hide()
			self.play_button.show()
			self.fps_spinBox.show()
			self.label.show()			
		elif mode == 'Pictures':
			source = self.open_file_dialog()
			self.prev_button.show()
			self.next_button.show()
			self.play_button.hide()
			self.fps_spinBox.hide()
			self.label.hide()
		elif mode == 'Web Camera':
			source = 'webCam0'
			self.prev_button.hide()
			self.next_button.hide()
			self.play_button.hide()
			self.fps_spinBox.hide()
			self.label.hide()
		
	
		
		
class OwnImageWidget(QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


#global functions

def c (text, replace_last = False):
	console_queue.put([text, replace_last])
		
def capture_and_detection_daemon():	
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
		with tf.Session(graph=detection_graph) as sess:
			c('Detection session has began...')
			while(True):
				global source, mode, player_queue, capture, min_score, RAC_qty
				if source:
					if mode == 'Video file':
						c('Source: {0}'.format(str(source)))
						capture = cv2.VideoCapture(source)
						capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
						capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
						capture.set(cv2.CAP_PROP_FPS, 25)
						length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
						player_queue = queue.Queue()
						frames_processed = 0
						while(True):
							frame = {}
							try:								
								retval, image_np = capture.read(0)
								frame["img"] = image_np
								frames_processed += 1
							except cv2.error as e:
								c('OpenCV error occured')
							if retval:
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
								(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})

								# Visualization of the results of a detection.
								scores = np.squeeze(scores)
								boxes = np.squeeze(boxes)
								classes = np.squeeze(classes).astype(np.int32)
								vis_util.visualize_boxes_and_labels_on_image_array(
								image_np,
								boxes,
								classes,
								scores,
								category_index,
								use_normalized_coordinates=True,
								min_score_thresh = min_score,
								line_thickness=8)
								score = np.amax(scores)
								frame = {}
								#print(str(image_np[3]))
								frame["img"] = image_np
								frame["score"] = score
								player_queue.put(frame)
							else:
								c('Recognition done...')
								capture.release()
								source = None
								player_list.append(player_queue)
								break
							#c('{0}% processed...'.format(round(100*frames_processed/length, 1)), True)
					elif mode == 'Pictures':
						frames_processed = 0
						length = len(source)
						player_queue = queue.Queue()
						w.video.update()
						for picture in source:
							c('Source image: {0}'.format(picture))
							capture = cv2.VideoCapture(picture)
							capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
							capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
							frame = {}
							try:								
								retval, image_np = capture.read(0)
								capture.release()
								frame["img"] = image_np
								frames_processed += 1
							except cv2.error as e:
								c('OpenCV error occured')
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
							(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
							# Visualization of the results of a detection.
							vis_util.visualize_boxes_and_labels_on_image_array(
							image_np,
							np.squeeze(boxes),
							np.squeeze(classes).astype(np.int32),
							np.squeeze(scores),
							category_index,
							use_normalized_coordinates=True,
							line_thickness=8)							
							frame = {}        
							frame["img"] = image_np
							player_queue.put(frame)
							create_snippet(image_np, np.squeeze(boxes), np.squeeze(scores), picture)
						c('Recognition done...')
						source = None
						
					elif mode == 'Web Camera':
						player_queue = queue.Queue()
						capture = cv2.VideoCapture(0)
						if capture is None or not capture.isOpened():
							c('No web camera!', True)
						else:
							capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
							capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
							capture.set(cv2.CAP_PROP_FPS, 25)
							while(True):
								frame = {}
								try:								
									retval, image_np = capture.read(0)
									frame["img"] = image_np
								except cv2.error as e:
									c('OpenCV error occured')	
								if retval:
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
									(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})

									# Visualization of the results of a detection.
									vis_util.visualize_boxes_and_labels_on_image_array(
									image_np,
									np.squeeze(boxes),
									np.squeeze(classes).astype(np.int32),
									np.squeeze(scores),
									category_index,
									use_normalized_coordinates=True,
									line_thickness=8)

									frame = {}        
									frame["img"] = image_np
									player_queue.put(frame)
									if not capture:
										c('break!')
										break
						
						
					time.sleep(0.1)
				elif player_queue.empty() and not source:
					c('No source...', True)
					time.sleep(2)			
				elif not source and player_queue.empty():
					c('No source...', True)
				
def create_snippet(image, boxes, scores, image_name, max_boxes_to_draw=1, min_score_thresh=.5, agnostic_mode=False):	
	box_to_display_str_map = collections.defaultdict(list)
	box_to_color_map = collections.defaultdict(str)
	if not max_boxes_to_draw:
		max_boxes_to_draw = boxes.shape[0]
	for i in range(min(max_boxes_to_draw, boxes.shape[0])):
		if scores is None or scores[i] > min_score_thresh:
			box = tuple(boxes[i].tolist())
			box_to_color_map[box] = 'Not_needed'
	for box, color in box_to_color_map.items():
		if box:
			ymin, xmin, ymax, xmax = box
			img_height, img_width, channels_qty = image.shape
			ymin_px = int(round(img_height*ymin, 0))
			xmin_px = int(round(img_height*xmin, 0))
			ymax_px = int(round(img_height*ymax, 0))
			xmax_px = int(round(img_height*xmax, 0))
			xcenter_px = int(round(((xmax-xmin)/2+xmin)*img_width, 0))
			ycenter_px = int(round(((ymax-ymin)/2+ymin)*img_height, 0))
			frame_width_px = xmax_px - xmin_px
			frame_height_px = ymax_px - ymin_px
			c('RAC detected: xmin = {1}, ymin = {0}, xmax = {3}, ymax = {2}'.format(ymin_px, xmin_px, ymax_px, xmax_px))
			path = image_name[:image_name.rfind('/')+1]
			if not os.path.exists(path + 'output'):
				os.mkdir(path + 'output')
			file_name = image_name[image_name.rfind('/')+1:-3] + 'json'
			data = json.dumps(
				{
				'Team': 'EA^2 (Sudakov)',
				'ImageName': image_name[image_name.rfind('/')+1:],
				'Region': 
					{
					'Main':
						{
						'TopLeft':[xmin_px, ymin_px],
						'BotRight':[xmax_px, ymax_px]
						},
					'Alternative':
						{
						'Center':[xcenter_px, ycenter_px],
						'Width':frame_width_px,
						'Height':frame_height_px
						},
					'EachPoint':
						[
							{
							'Comment':'top-left, (x; y)',
							'Point':[xmin_px, ymin_px]
							},
							{
							'Comment':'top-right, (x; y)',
							'Point':[xmax_px, ymin_px]
							},
							{
							'Comment':'bottom-right, (x; y)',
							'Point':[xmax_px, ymax_px]
							},
							{
							'Comment':'bottom-left, (x; y)',
							'Point':[xmin_px, ymax_px]
							}
						]
					}
				}, 
				indent=4)
			with open(path + 'output/' + file_name, 'w', encoding='utf8') as file:
				file.write(data)
				file.close()
			c('File created: {0}'.format(file_name))
		
	
	
#main

if __name__ == '__main__':    
	app = QApplication(sys.argv)
	w = RAC_app_window(None)
	w.setWindowTitle('RAC_app')
	w.show()
	
app.exec_()