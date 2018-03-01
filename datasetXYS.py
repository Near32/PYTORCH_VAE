import xml.etree.ElementTree as ET
import numpy as np
import os 
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

import cv2


class RandomRecolorNormalize(object) :
	def __init__(self,sizew=224,sizeh=224) :
		self.sizeh = sizeh
		self.sizew = sizew

	def __call__(self,sample) :
		img, gaze = sample['image'], sample['gaze']
		h,w,c = img.shape

		
		#recolor :
		t = [np.random.uniform()]
		t += [np.random.uniform()]
		t += [np.random.uniform()]
		t = np.array(t)

		img = img * (1+t)

		# Normalize color between 0 and 1 :
		img = img / (255.0*1.0)

		# Normalize the size of the image :
		#img = cv2.resize(img, (self.sizeh,self.sizew))

		return {'image':img, 'gaze':gaze}


class data2loc(object) :
	def __call__(self,sample) :
		img, gaze = sample['image'], sample['gaze']
		h,w,c = img.shape

		outputs = np.zeros((1,2))

		outputs[0,0] = gaze['x']
		outputs[0,1] = gaze['y']
				
			
		return {'image':img, 'outputs':outputs}


class ToTensor(object) :
	def __call__(self, sample) :
		image, outputs = sample['image'], sample['outputs']
		#swap color axis :
		# numpy : H x W x C
		# torch : C x H x W
		image = image.transpose( (2,0,1) )
		return {'image':torch.from_numpy(image/255.0), 'landmarks':torch.from_numpy(outputs) }

Transform = transforms.Compose([
							data2loc(),
							ToTensor()
							])

TransformPlus = transforms.Compose([
							RandomRecolorNormalize(),
							data2loc(),
							ToTensor()
							])


def parse_annotation_GazeRecognition(ann_dir) :
	imgs = []

	for ann in os.listdir(ann_dir) :
		img = {}

		tree = ET.parse(os.path.join(ann_dir,ann) )


		for elem in tree.iter() :
			if 'filename' in elem.tag :
				imgs += [img]
				img['filename'] = elem.text

			if 'width' in elem.tag :
				img['width'] = int(float(elem.text))
			if 'height' in elem.tag :
				img['height'] = int(float(elem.text))
			
			if 'data' in elem.tag:
				data = {}
				img['data'] = data
				
				for attr in list(elem) :
					if 'model' in attr.tag :
						data['model'] = attr.text
					if 'gaze_position' in attr.tag :
						gaze = {}
						data['gaze'] = gaze
						
						for attri in list(attr) :
							if 'x' in attri.tag :
								gaze['x'] = float(attri.text)
							if 'y' in attri.tag :
								gaze['y'] = float(attri.text)
					if 'screen_size' in attr.tag :
						screen = {}
						data['screen'] = screen
						
						for attri in list(attr) :
							if 'width' in attri.tag :
								screen['width'] = float(attri.text)
							if 'height' in attri.tag :
								screen['height'] = float(attri.text)
					if 'camera_screen' in attr.tag :
						cam_screen = {}
						data['camera_screen_center_offset'] = cam_screen
						
						for attri in list(attr) :
							if 'x' in attri.tag :
								cam_screen['x'] = float(attri.text)
							if 'y' in attri.tag :
								cam_screen['y'] = float(attri.text)
					if 'head' in attr.tag :
						head = {}
						data['head'] = head

						for attri in list(attr) :
							if 'head_camera_distance' in attri.tag :
								head['head_camera_distance'] = float(attri.text)

			if 'object' in elem.tag:
				name = None
				bndbox = [0,0,0,0]

				for attr in list(elem) :
					if 'name' in attr.tag :
						name = attr.text
					if 'bndbox' in attr.tag :
						for attri in list(attr) :
							if 'xmin' in attri.tag :
								bndbox[0] = float(attri.text)
							if 'ymin' in attri.tag :
								bndbox[1] = float(attri.text)
							if 'xmax' in attri.tag :
								bndbox[2] = float(attri.text)
							if 'ymax' in attri.tag :
								bndbox[3] = float(attri.text)
				
				if name is not None :
					img[name] = bndbox
				
						
					
	return imgs



class DatasetGazeRecognition(Dataset) :
	def __init__(self,img_dir,ann_dir,width=224,height=224,transform=TransformPlus,stacking=False,divide2=False):
		super(DatasetGazeRecognition,self).__init__()
		self.img_dir = img_dir
		self.ann_dir = ann_dir
		self.stacking = stacking
		self.divide2 = divide2

		self.w = width
		self.h = height

		self.parsedAnnotations = parse_annotation_GazeRecognition(self.ann_dir)

		self.transform = transform
		#default transformations :
		# ...
		# -2 : data2loc : transform the data list of dictionnaries into usable numpy outputs  
		# -1 : ToTensor

		self.idxModels = dict()
		for idx in range( len(self) ) :
			model = self.parsedAnnotations[idx]['data']['model']
			if not( model in self.idxModels.keys() ) :
				self.idxModels[model] = list()
			self.idxModels[model].append(idx)

		for model in self.idxModels.keys() :
			print('Model : {} :: {} pictures.'.format(model, len(self.idxModels[model]) ) )


	def __len__(self) :
		return len(self.parsedAnnotations)

	def __getitem__(self,idx) :
		path = os.path.join(self.img_dir,self.parsedAnnotations[idx]['filename']+'.png' )
		img = cv2.imread(path)
		h,w,c = img.shape 
		
		if self.stacking :
			img = np.expand_dims( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2)

			face_bndbox = self.parsedAnnotations[idx]['face']
			reye_bndbox = self.parsedAnnotations[idx]['reye']
			leye_bndbox = self.parsedAnnotations[idx]['leye']

			scalar = 1.0
			if self.divide2 :
				scalar = 2.0
						
			# face :
			fy1 = int( min( max(0,face_bndbox[1]/scalar), h) )
			fy2 = int( min( max(0,face_bndbox[3]/scalar), h) )
			fx1 = int( min( max(0,face_bndbox[0]/scalar), w) )
			fx2 = int( min( max(0,face_bndbox[2]/scalar), w) )
			
			face_img = img[fy1:fy2, fx1:fx2,:]
			face_img = np.expand_dims(  cv2.resize(face_img, (w,h) ), 2)
			
			# reye :
			ry1 = int( min( max(0,reye_bndbox[1]/scalar), h) )
			ry2 = int( min( max(0,reye_bndbox[3]/scalar), h) )
			rx1 = int( min( max(0,reye_bndbox[0]/scalar), w) )
			rx2 = int( min( max(0,reye_bndbox[2]/scalar), w) )
			
			reye_img = img[ry1:ry2, rx1:rx2,:]
			reye_img = np.expand_dims( cv2.resize(reye_img, (w,h) ), 2)
			
			# leye :
			ly1 = int( min( max(0,leye_bndbox[1]/scalar), h) )
			ly2 = int( min( max(0,leye_bndbox[3]/scalar), h) )
			lx1 = int( min( max(0,leye_bndbox[0]/scalar), w) )
			lx2 = int( min( max(0,leye_bndbox[2]/scalar), w) )
			
			leye_img = img[ly1:ly2, lx1:lx2,:]
			leye_img = np.expand_dims( cv2.resize(leye_img, (w,h) ), 2)
			
			# concatenation :
			img = np.concatenate( [img, reye_img, leye_img], axis=2)

		img = np.ascontiguousarray(img)
		img = cv2.resize( img, (self.h, self.w) )

		gaze = copy.deepcopy(self.parsedAnnotations[idx]['data']['gaze'])
		cam_screen_offset = copy.deepcopy(self.parsedAnnotations[idx]['data']['camera_screen_center_offset'])
		for el in ['x','y'] :
			gaze[el] += cam_screen_offset[el]

		sample = {'image':img, 'gaze':gaze}

		if self.transform is not None :
			sample = self.transform(sample)

		return sample

	def generateVisualization(self, idx, shape=None, ratio=30, screen_size=[0.12,0.05],estimation=[0.02,0.02], cm_prec=0.02) :
		idx = int(idx)
		try :
			path = os.path.join(self.img_dir,self.parsedAnnotations[idx]['filename']+'.png' )
			img = cv2.imread(path)
			img = np.ascontiguousarray(img)
			
			gaze = copy.deepcopy(self.parsedAnnotations[idx]['data']['gaze'])
			cam_screen_offset = copy.deepcopy(self.parsedAnnotations[idx]['data']['camera_screen_center_offset'])
			for el in ['x','y'] :
				gaze[el] += cam_screen_offset[el]

			if shape is None :
				shape = list(img.shape)
			else :
				img = cv2.resize( img, shape)

			h,w,d = img.shape
			img = cv2.resize( img, (self.h, self.w) )
			# create visualization :
			visualization = 255*np.ones( (480,640,3), dtype=np.float32 )
			ratio = 640/(2*screen_size[1]*100)
			px_screen_size = np.array(screen_size) * 100 * ratio
			cam_offset = [-0.01,0.01]
			px_cam_offset = np.array(cam_offset) * 100 * ratio
			def draw_screen_cam(image,px_screen_size, px_cam_offset) :
				shape = np.array(image.shape)[0:2]
				offset = 10
				px_screen_offset = (shape - px_screen_size )/ 2
				pt1 = px_screen_offset
				pt2 = pt1+px_screen_size
				color = (0,0,0)

				pt1_t = (int(pt1[1])+offset,int(pt1[0]))
				pt2_t = (int(pt2[1])+offset,int(pt2[0]))
				cv2.rectangle(image, pt1_t, pt2_t, color=color, thickness=3)
				
				pt3 = pt1+px_cam_offset
				pt3_t = (int(pt3[1])+offset,int(pt3[0]))
				cv2.circle(image, pt3_t, radius=10, color=color, thickness=3)
				
				return image, pt3
			visualization, px_cam_pt = draw_screen_cam(visualization,px_screen_size,px_cam_offset)

			px_pt = np.array([ gaze['y'], gaze['x'] ]) * 100 * ratio
			px_estimation_pt = np.array(estimation) * 100 * ratio

			prec = int(cm_prec * 100 * ratio) 
			# 2 centimeter precision
			def draw_point(image,px_pt,prec,px_cam_pt,color=(255,255,255)) :
				pt = px_cam_pt+px_pt
				pt_t = ( int(pt[1]), int(pt[0]) )
				cv2.circle(image, pt_t, radius=prec, color=color, thickness=2)
				return image
			color_true = (0,255,0)
			visualization = draw_point(visualization,px_pt=px_pt,prec=4,px_cam_pt=px_cam_pt,color=color_true)
			color_est = (255,255,0)
			visualization = draw_point(visualization,px_pt=px_estimation_pt,prec=prec,px_cam_pt=px_cam_pt,color=color_est)

		except Exception as e :
			print(e)
			
		#if self.transform is not None :
		#	image = self.transform(image)
		#image = np.concatenate([image,visualization], axis=1)
		
		sample = {'image': img, 'visualization':visualization, 'gaze':gaze }
		
		return sample


class LinearClassifier(nn.Module) :
	def __init__(self, input_dim=10, output_dim=3) :
		super(LinearClassifier,self).__init__()

		self.input_dim = input_dim
		self.output_dim = output_dim

		self.fc = nn.Linear(self.input_dim,self.output_dim)

	def forward(self,x) :
		out = self.fc(x)
		soft_out = F.softmax(out)

		return soft_out


def test_stacking() :
	dataset = load_dataset_XYS(stacking=True)

	sample = dataset[0]

	img = sample['image']
	img0 = img[:,:,:].numpy().reshape((-1,224))

	while True :
		cv2.imshow('test',img0 )

		key = cv2.waitKey(30)
		if key == ord('q') :
			break


def test_dataset_visualization() :
	#ann_dir = '/media/kevin/Data/DATASETS/XYS-latent/annotations'
	#img_dir = '/media/kevin/Data/DATASETS/XYS-latent/images'
	ann_dir = './dataset-XYS-latent/annotations'
	img_dir = './dataset-XYS-latent/images'
	width = 448
	height = 448
	transform = TransformPlus

	dataset = DatasetGazeRecognition(img_dir=img_dir,ann_dir=ann_dir,width=width,height=height,transform=transform)

	i=0
	continuer = True
	screen_size = [0.20,0.20]
	while continuer :
		sample = dataset.generateVisualization(idx=0+i,screen_size=screen_size)

		cv2.imshow('image', sample['image'] )
		cv2.imshow('screen', sample['visualization'] )
		
		while True :
			key = cv2.waitKey()
			if key == ord('n'):
				i+=1
				break
			if key == ord('q'):
				continuer = False
				break


def load_dataset_XYS(img_dim=224,stacking=False) :
	#ann_dir = '/media/kevin/Data/DATASETS/XYS-latent/annotations'
	#img_dir = '/media/kevin/Data/DATASETS/XYS-latent/images'
	#ann_dir = '/home/kevin/DATASETS/dataset-XYS-latent/annotations'
	#img_dir = '/home/kevin/DATASETS/dataset-XYS-latent/images'
	ann_dir = './dataset-XYS-latent/annotations'
	img_dir = './dataset-XYS-latent/images'
	width = img_dim
	height = img_dim
	transform = Transform #TransformPlus

	datasets = DatasetGazeRecognition(img_dir=img_dir,ann_dir=ann_dir,width=width,height=height,transform=transform, stacking=stacking, divide2=True)
	
	return datasets


def load_dataset_XYSM10(img_dim=224,stacking=False) :
	ann_dir = './dataset-XYSM10-latent/annotations'
	img_dir = './dataset-XYSM10-latent/images'
	width = img_dim
	height = img_dim
	transform = Transform #TransformPlus

	datasets = DatasetGazeRecognition(img_dir=img_dir,ann_dir=ann_dir,width=width,height=height,transform=transform, stacking=stacking, divide2=True)
	
	return datasets


def generateIDX(dataset) :
	from math import floor
	nbrel = len(dataset.parsedAnnotations)
	gazex = [ round(dataset.parsedAnnotations[i]['data']['gaze']['x'], 3) for i in range(nbrel)  ]
	setgx = set(gazex)
	idx_gaze_x = [ [ idx for idx in range(nbrel) if gazex[idx] == gx] for gx in setgx]

	gazey = [  dataset.parsedAnnotations[i]['data']['gaze']['y'] for i in range(nbrel)  ]
	setgy = set(gazey)
	#print( len(setgy) )
	'''
	prec = 1e2
	gazeyf = [ floor( dataset.parsedAnnotations[i]['data']['gaze']['y']*prec)/prec for i in range(nbrel)  ]
	'''
	nbrval = 10
	limit = 0.349
	step = limit/nbrval
	ceil_vals = []
	val = 0.0
	for i in range(nbrval+1) :
		val += step
		ceil_vals.append( val)
	#print(ceil_vals)
	#print(len(ceil_vals))

	idx_gaze_y = list()
	for i in range(nbrval+1) :
		idx_gaze_y.append( list() )
	
	for i in range(len(gazey) ) :
		idx_ceil = 0 
		while ceil_vals[idx_ceil] <= gazey[i] :
			idx_ceil += 1
		idx_gaze_y[idx_ceil].append( i)

	'''
	print(idx_gaze_y[0])
	for i in range(nbrval) :
		print( len(idx_gaze_y[i]) ) 	
	'''
	'''
	for i in idx_gaze_y[0] :
		print( ' idx: {}  ::  {} >= {}'.format( i, ceil_vals[0], gazey[ i ]) )	
	'''

	headd = [ dataset.parsedAnnotations[i]['data']['head']['head_camera_distance'] for i in range(nbrel)  ]
	sethdd = set(headd)
	#print( len(sethdd) )
	#print(sethdd)
	idx_head_distance = [ [ idx for idx in range(nbrel) if headd[idx] == hdd] for hdd in sethdd]

	return idx_gaze_x, idx_gaze_y[0:10], idx_head_distance


def generateClassifier(input_dim=10,output_dim=3) :
	return LinearClassifier(input_dim=input_dim,output_dim=output_dim)


def test() :
	dataset = load_dataset_XYS(img_dim=128)
	idxgx, idxgy, idxhead = generateIDX(dataset)

	print( len(idxgx) )
	print( len(idxgy) )
	print( len(idxhead) )


if __name__ == '__main__' :
	#test_dataset()
	#test_dataset_visualization()
	test_stacking()
	#test()
