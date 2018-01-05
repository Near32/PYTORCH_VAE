import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image 

class dSpriteDataset(Dataset) :
	def __init__(self, root='./dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', transform=None) :
		self.root = root
		self.transform = transform

		# Load dataset
		dataset_zip = np.load(self.root)
		print('Keys in the dataset:', dataset_zip.keys())
		self.imgs = dataset_zip['imgs']
		self.latents_values = dataset_zip['latents_values']
		self.latents_classes = dataset_zip['latents_classes']
		#self.metadata = dataset_zip['metadata'][()]
		#print('Metadata: \n', metadata)
		print('Dataset loaded : OK.')

	def __len__(self) :
		return len(self.imgs)

	def __getitem__(self, idx) :
		image = Image.fromarray(self.imgs[idx])
		latent = self.latents_values[idx]
		
		if self.transform is not None :
			image = self.transform(image)

		sample = (image, latent)

		
		return sample

def test_dSprite() :
	import cv2
	
	root = './dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
	dataset = dSpriteDataset(root=root)

	idx = 0
	
	while True :
		
		sample = dataset[idx]
		img = np.array( sample[0])*255
		cv2.imshow('test',img)
		
		key = cv2.waitKey(1) & 0xFF
		if  key == ord('q'):
			break
		elif key == ord('n') :
			idx += 1
			print("next image")


if __name__ == "__main__" :
	test_dSprite()

