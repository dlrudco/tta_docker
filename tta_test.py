import sys
sys.path.append('/home/ncl/models/research')
sys.path.append('/home/ncl/models/research/slim')
import feature
from feature import Compare
import cv2

import time
import os

import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt

import pickle
import csv
import argparse 
import time

compmgr = Compare()


def read_all_vectors(scene_folder, debug=True): ## Perform vectorization also when reading images
	scene_folder_path = scene_folder
	video_folders = os.listdir(scene_folder)
	if debug:
		print(scene_folder_path)
	
	assert len(video_folders) >= 1, 'Not enough subfolders for scene folder:'+ scene_folder_path
	if os.path.isfile(os.path.join(scene_folder, 'data.pkl')):
		with open(os.path.join(scene_folder, 'data.pkl'), 'rb') as f:
			datadict = pickle.load(f)
	else:
		datadict = {}
		for i, video_folder in enumerate(video_folders):
			video_folder_path = os.path.join(scene_folder_path, video_folder)
			if not os.path.isdir(video_folder_path):
				continue
			if debug:
				print('\t'+video_folder)
			car_folders = os.listdir(video_folder_path)

			assert len(car_folders) >= 1, 'Not enough subfolders for video folder:'+ video_folder_path
			datadict[video_folder] = {}

			for j, car_folder in enumerate(car_folders):
				if debug:
					st = time.time()
					print('\t\t'+car_folder)
				car_folder_path = os.path.join(video_folder_path, car_folder)
				if not os.path.isdir(car_folder_path):
					continue
				images = os.listdir(car_folder_path)

				# assert len(images) >= 1, 'Not enough images for car folder:'+ car_folder_path
				if len(images) == 0:
					continue
				pathlist = []
				imlist = []
				veclist = []
				wveclist = []
				for k, image_file in enumerate(images):
					if not image_file.split('.')[-1] == 'png':
						continue
					image_path = os.path.join(car_folder_path, image_file)
					roi_path = image_path.replace('png','csv')
					image = cv2.imread(image_path)
					assert image.size > 0, 'Image not read for ' + image_path
					roi_file = open(roi_path)
					roi = list(map(int,map(float,csv.reader(roi_file).__next__())))
					bbox = [roi[1], roi[0], roi[1]+roi[3], roi[0]+roi[2]]
					crop_image = image[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
					vector = compmgr.ext_delf_feature(crop_image,whole_image=True)
					if vector[0][0].shape == (0,):
						continue
					veclist.append((vector))
					# wveclist.append((compmgr.ext_delf_feature(image,whole_image=True)))
					imlist.append(image)
					pathlist.append(image_path)
					# cv2.imshow('.', crop_image)
					# cv2.waitKey(0)
				if not veclist == []:
					datadict[video_folder][car_folder] = {'Images' : imlist, 'Paths' : pathlist, 'Vectors' : veclist, 'wVectors' : wveclist}
				else:
					print("Vector Not extracted. Please check if there is an image inside the folder {}".format(car_folder_pat))
				if debug:
					# print(datadict[video_folder][car_folder])
					print('\t\t\t\tTook {} seconds for {} images'.format(time.time()-st, len(images)//2))
		with open(os.path.join(scene_folder, 'data.pkl'), 'wb') as f:
			pickle.dump(datadict,f)


	return datadict


def filterByKey(data,keys):
	return {x: data[x] for x in keys}

def sigmoid(x):
    return 1 / (1 +np.exp(-x))


def test_2020(vector1, vector2, same):
	simmat = compmgr.ComputeSimilarity_all(vector1[0],vector2[0],vector1[1],vector2[1])
	# wsimmat = compmgr.ComputeSimilarity_all(wvector1[0],wvector2[0],wvector1[1],wvector2[1])
	# print(simmat)
	try:
		_ = simmat.max()
	except ValueError:
		return -1, None
	if simmat.max() == -1:
		print(simmat)
		return -1, None
	# print(str(time.time()-start))
	perc = simmat.max()
	if same:
		return 1, perc
	else:
		return 0, perc
	

def roc_same(dic, videos, cars, label, pred):
	count = 0
	tcount = 0
	same = True
	tcount = len(dic[videos[0]][cars[0]]['Vectors']) * len(dic[videos[1]][cars[1]]['Vectors'])

	for i, vector_A in enumerate(dic[videos[0]][cars[0]]['Vectors']):
		for j, vector_B in enumerate(dic[videos[1]][cars[1]]['Vectors']):
			y_true, y_pred = test_2020(vector_A, vector_B, same)
			if y_true == -1:
				raise ValueError
			label.append(y_true)
			pred.append(y_pred)
			count += 1
			print(f"\r({count}/{tcount})" + " Image Processing Done!!", sep=' ', end='', flush=True)
	print("")
	return label, pred, count


def roc_diff(dic, videos, cars, label, pred):
	count = 0
	tcount = 0
	same = False
	tcount = len(dic[videos[0]][cars[0]]['Vectors']) * len(dic[videos[1]][cars[1]]['Vectors'])

	for i, vector_A in enumerate(dic[videos[0]][cars[0]]['Vectors']):
		for j, vector_B in enumerate(dic[videos[1]][cars[1]]['Vectors']):
			y_true, y_pred = test_2020(vector_A, vector_B, same)
			if y_true == -1:
				raise ValueError
			label.append(y_true)
			pred.append(y_pred)
			count += 1
			print(f"\r({count}/{tcount})" + " Image Processing Done!!", sep=' ', end='', flush=True)
	print("")
	return label, pred, count


def roc_eval(data_dict):
	label = []
	pred = []
	samecount = 0
	diffcount = 0
	videos = list(data_dict.keys())

	assert len(videos) == 2

	for car_A in data_dict[videos[0]].keys():
		for car_B in data_dict[videos[1]].keys():
			if car_A == car_B:
				print('{} : Same'.format(car_A))
				label, pred, count = roc_same(data_dict, videos, [car_A, car_B], label, pred)
				samecount += count
			else:
				print('{}-{} : Different'.format(car_A, car_B))
				label, pred, count = roc_diff(data_dict, videos, [car_A, car_B], label, pred)
				diffcount += count
	return label, pred, samecount, diffcount



def main(args, scene_folder, data_dict):
	if os.path.isfile('{}/label.pkl'.format(scene_folder)):
		with open('{}/label.pkl'.format(scene_folder), 'rb') as labelpkl:
			label = pickle.load(labelpkl)
		with open('{}/pred.pkl'.format(scene_folder), 'rb') as predpkl:
			pred = pickle.load(predpkl)
	else:
		label, pred, samecount, diffcount = roc_eval(data_dict)
			
		labelpkl = open('{}/label.pkl'.format(scene_folder), 'wb')
		pickle.dump(label, labelpkl)
		labelpkl.close()

		predpkl = open('{}/pred.pkl'.format(scene_folder), 'wb')
		pickle.dump(pred, predpkl)
		predpkl.close()



	fpr, tpr, thres = roc_curve(label, pred)

	roc_auc = auc(fpr, tpr)
	print("Total test cases : "+str(len(label)))
	print('ROC-AUC: {0:0.2f}'.format(
      	roc_auc))
	#plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve and AUC for Vehicle Re-ID')
	plt.legend(loc="lower right")
	
	plt.savefig('{}/ROC.png'.format(scene_folder), dpi=300)
	if args.show:
		plt.show()

	


	#pr, rec, thres = precision_recall_curve(label, pred)
	#pr_auc = auc(rec, pr)
	#average_precision = average_precision_score(label, pred)

	#print('Average precision-recall score: {0:0.2f}'.format(
      	#average_precision))
	#plt.figure()
	#lw = 2
	#plt.plot(rec, pr, color='darkorange',
	#         lw=lw, label=f'PR curve (area = {pr_auc}, AP = {average_precision})')
	#plt.xlim([0.0, 1.0])
	#plt.ylim([0.0, 1.05])
	#plt.xlabel('Recall')
	#plt.ylabel('Precision')
	#plt.title('PR curve and AUC for Vehicle Re-ID')
	#plt.legend(loc="lower right")
	
	#plt.savefig('{}/PR.png'.format(scene_folder), dpi=300)
	#if args.show:
	#	plt.show()


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='TTA test code for Smart City')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose mode, printing out logs if given')
    parser.add_argument('-s', '--show', dest='show', action='store_true', help='Visualization, Shows ROC and PR curve in addition to saving the figure. Saving Figure is done anyway.')
    parser.add_argument('--scene', dest='scene_folder', nargs='+', help='Select the target scene folder for evaluation')
    
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
	args = parse_args()
	print(args)
	for scene_index, scene_folder in enumerate(args.scene_folder):
		st = time.time()
		data_dict = read_all_vectors(scene_folder, debug=args.verbose)
		main(args, scene_folder, data_dict)
		print('Took {} seconds for scene {}'.format(time.time()-st, scene_folder))
