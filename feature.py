from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import cv2
from scipy import signal
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from PIL import ImageFile
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import delf_config_pb2
from delf import aggregation_config_pb2
from delf import datum_io
from delf import feature_aggregation_extractor
from delf import feature_io
from delf.python.detect_to_retrieve import dataset
from delf import extractor
from delf import feature_aggregation_similarity


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

CONFIG_ROOT = "/home/ncl/kclee/sc2020/scdelf/configs/"

_DELF_EXTENSION = '.delf'
_IMAGE_EXTENSION = '.jpg'
_VLAD_EXTENSION_SUFFIX = 'vlad'
_ASMK_EXTENSION_SUFFIX = 'asmk'
_ASMK_STAR_EXTENSION_SUFFIX = 'asmk_star'


_VLAD = aggregation_config_pb2.AggregationConfig.VLAD
_ASMK = aggregation_config_pb2.AggregationConfig.ASMK
_ASMK_STAR = aggregation_config_pb2.AggregationConfig.ASMK_STAR


ImageFile.LOAD_TRUNCATED_IMAGES = True

def _PilLoader_old(path):
  with tf.gfile.GFile(path, 'rb') as f:
    img = Image.open(f)
    # print(img.size)
    img = img.resize((int(img.size[0]/2),int(img.size[1]/2)))
    # print(img.size)
    return img.convert('RGB')

def _PilLoader(path, resize=True, ratio=2):
	img = Image.open(path)
	# print(img.size)
	# img = img.resize((672,504))
	if resize:
		if img.size[0] > img.size[1]:
			img = img.rotate(-90)
		img = img.resize((int(img.size[0]/ratio),int(img.size[1]/ratio)))
	
	# print(img.size)
	return img.convert('RGB')


class Compare:
	def __init__(self):
		# self.detector = Detector()
		self.delf_config = delf_config_pb2.DelfConfig()

		with tf.gfile.GFile(os.path.join(CONFIG_ROOT,'delf_gld_config.pbtxt'), 'r') as f:
			text_format.Merge(f.read(), self.delf_config)
		self.delf_graph = tf.Graph()
		self.delf_sess = tf.Session(graph=self.delf_graph)

		
		with self.delf_graph.as_default():
			init_op = tf.compat.v1.global_variables_initializer()
			self.delf_sess.run(init_op)
			self.delf_extractor = extractor.MakeExtractor(self.delf_sess, self.delf_config)


		self.agg_config = aggregation_config_pb2.AggregationConfig()
		with tf.gfile.GFile(os.path.join(CONFIG_ROOT,'roadcar_query.pbtxt'), 'r') as f:
			text_format.Merge(f.read(), self.agg_config)
		output_extension = '.'
		if self.agg_config.aggregation_type == _VLAD:
			output_extension += _VLAD_EXTENSION_SUFFIX
		elif self.agg_config.aggregation_type == _ASMK:
			output_extension += _ASMK_EXTENSION_SUFFIX
		elif self.agg_config.aggregation_type == _ASMK_STAR:
			output_extension += _ASMK_STAR_EXTENSION_SUFFIX
		else:
			raise ValueError('Invalid aggregation type: %d' % config.aggregation_type)

		self.agg_graph = tf.Graph()
		self.agg_sess = tf.Session(graph=self.agg_graph)

		with self.agg_graph.as_default():
			init_op = tf.compat.v1.global_variables_initializer()
			self.agg_sess.run(init_op)
			self.agg_extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
        														self.agg_sess, self.agg_config)
			self.similarity_computer = (feature_aggregation_similarity.SimilarityAggregatedRepresentation(
									self.agg_config))


	def ext_delf_feature(self,image,item=None,whole_image=False, for_demo=False):
		# Crop query image according to bounding box.
		imlist = []
		desclist = []
		wordlist = []
		bbox = []
		# st = time.time()
		if whole_image:
			im= np.array(image)
			(locations_out, descriptors_out, feature_scales_out,
				attention_out) = self.delf_extractor(im)
			if not descriptors_out.shape[0]:
				descriptors_out = np.reshape(descriptors_out,
			                       [0, self.agg_config.feature_dimensionality])
			num_features_per_box = None


			(aggregated_descriptors,
			feature_visual_words) = self.agg_extractor.Extract(descriptors_out,
			                                     num_features_per_box)

			desclist.append(aggregated_descriptors)
			wordlist.append(feature_visual_words)

			return desclist, wordlist
		else:
			raise ValueError("Detection Part is not yet planted")
		# else:
		# 	try:
		# 		res, scale = self.detector.forward(image)
		# 	except TypeError:
		# 		image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
		# 		res, scale = self.detector.forward(image)
		# 	i = 7
		# 	j = 0
		# 	flag = False
		# 	best_size = 0
		# 	obj_index = 0
		# 	while res[0, i, j, 0] >= 0.6:
		# 		pt = list(map(int,(np.clip(res[0, i, j, 1:].cpu().numpy(),0.0,1.0)*scale.cpu().numpy())))
		# 		coords = [pt[0], pt[1], pt[2], pt[3]]
		# 		size = (pt[2]-pt[0])*(pt[3]-pt[1])
		# 		if size > best_size:
		# 			best_size = size
		# 		j+=1
		# 	j=0
		# 	while res[0, i, j, 0] >= 0.6:
		# 		flag = True
		# 		# print(res[0, i, j, 0])
		# 		score = res[0, i, j, 0]
		# 		pt = list(map(int,(np.clip(res[0, i, j, 1:].cpu().numpy(),0.0,1.0)*scale.cpu().numpy())))
		# 		coords = [pt[0], pt[1], pt[2], pt[3]]
		# 		size = (pt[2]-pt[0])*(pt[3]-pt[1])
		# 		if size != best_size:
		# 			j+=1
		# 			continue

		# 		xlen = pt[3] - pt[1]
		# 		ylen = pt[2] - pt[0]
		# 		j+=1	
		# 		try:
		# 			im= np.array(image.crop(coords))
		# 		except AttributeError:
		# 			im = image[pt[1]:pt[3],pt[0]:pt[2],:]
		# 		bbox = coords
		# 		imlist.append(im)

		# 		(locations_out, descriptors_out, feature_scales_out,
		# 		attention_out) = self.delf_extractor(im)


		# 		if not descriptors_out.shape[0]:
		# 			descriptors_out = np.reshape(descriptors_out,
		# 		                       [0, self.agg_config.feature_dimensionality])
		# 		num_features_per_box = None

		# 		(aggregated_descriptors,
		# 		feature_visual_words) = self.agg_extractor.Extract(descriptors_out,
		# 		                                     num_features_per_box)

		# 		desclist.append(aggregated_descriptors)
		# 		wordlist.append(feature_visual_words)
		# 	if for_demo:
		# 		if flag:	
		# 			return desclist, wordlist, bbox		
		# 		else:
		# 			raise ValueError('Could not detect any car instance in this image!!')
		# 	else:
		# 		if flag:	
		# 			return desclist, wordlist		
		# 		else:
		# 			raise ValueError('Could not detect any car instance in this image!!')

	def ComputeSimilarity_all(self, desclist1,desclist2,wordlist1,wordlist2):
		simmat = np.empty((len(desclist1),len(desclist2)))
		for i in range(len(desclist1)):
			for j in range(len(desclist2)):
				sim = self.similarity_computer.ComputeSimilarity(desclist1[i], desclist2[j], wordlist1[i], wordlist2[j])
				simmat[i][j] = sim

		return simmat

