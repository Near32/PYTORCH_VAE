import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms 

from skimage import io, transform
import numpy as np

import math
from PIL import Image


from models import Rescale, betaVAE, betaVAEdSprite, betaVAEXYS, betaVAEXYS2, betaVAEXYS3, Bernoulli, GazeHead
from datasetXYS import load_dataset_XYS

use_cuda = True


def setting(nbr_epoch=100,offset=0,train=True,batch_size=32, evaluate=False,stacking=False,lr = 1e-5,z_dim = 3, train_head=False):	
	size = 256
	dataset = load_dataset_XYS(img_dim=size,stacking=stacking)

	# Data loader
	data_loader = torch.utils.data.DataLoader(dataset=dataset,
    	                                      batch_size=batch_size, 
        	                                  shuffle=True)

	# Model :
	'''
	frompath = True
	img_dim = size
	img_depth=3
	conv_dim = 32
	global use_cuda
	net_depth = 5
	beta = 5000e0
	betavae = betaVAEXYS(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	'''
	'''
	frompath = True
	img_dim = size
	img_depth=3
	conv_dim = 8#32
	global use_cuda
	net_depth = 5
	beta = 1000e0
	betavae = betaVAEXYS2(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	'''
	frompath = True
	img_dim = size
	img_depth=3
	conv_dim = 8#32
	global use_cuda
	net_depth = 6
	beta = 1000e0
	betavae = betaVAEXYS3(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	print(betavae)

		
	# LOADING :

	path = 'test3--XYS--img{}-lr{}-beta{}-layers{}-z{}-conv{}'.format(img_dim,lr,beta,net_depth,z_dim,conv_dim)
	if stacking :
		path+= '-stacked'

	if not os.path.exists( './beta-data/{}/'.format(path) ) :
		os.mkdir('./beta-data/{}/'.format(path))
	if not os.path.exists( './beta-data/{}/gen_images/'.format(path) ) :
			os.mkdir('./beta-data/{}/gen_images/'.format(path))
	if not os.path.exists( './beta-data/{}/reconst_images/'.format(path) ) :
			os.mkdir('./beta-data/{}/reconst_images/'.format(path))
	
	
	SAVE_PATH = './beta-data/{}'.format(path) 

	if frompath :
		try :
			betavae.load_state_dict( torch.load( os.path.join(SAVE_PATH,'weights')) )
			print('NET LOADING : OK.')
		except Exception as e :
			print('EXCEPTION : NET LOADING : {}'.format(e) )

	# GAZE HEAD :
	if train_head :
		gazehead = GazeHead(outdim=2, nbr_latents=z_dim, use_cuda=use_cuda)

	# LOADING :
	gh_path = 'test3--XYS--img{}-lr{}-beta{}-layers{}-z{}-conv{}'.format(img_dim,lr,beta,net_depth,z_dim,conv_dim)
	if stacking :
		gh_path+= '-stacked'

	if not os.path.exists( './beta-data/{}/'.format(gh_path) ) :
		os.mkdir('./beta-data/{}/'.format(gh_path))
	gh_SAVE_PATH = os.path.join('./beta-data/{}'.format(path), 'gazehead.weights') 
	gazehead.setSAVE_PATH(gh_SAVE_PATH)
	
	try :
		gazehead.load_state_dict( torch.load( gh_SAVE_PATH) )
		print('GAZE HEAD NET LOADING : OK.')
	except Exception as e :
		print('EXCEPTION : GAZE HEAD NET LOADING : {}'.format(e) )	



	# OPTIMIZER :
	if not train_head :
		optimizer = torch.optim.Adam( betavae.parameters(), lr=lr)
	else :
		optimizers = dict()
		optimizers['model'] = torch.optim.Adam( betavae.parameters(), lr=lr)
		optimizers['head'] = torch.optim.Adam( gazehead.parameters(), lr=lr)

	if train :
		if train_head :
			train_model_head(betavae, gazehead, data_loader, optimizers, SAVE_PATH,path,nbr_epoch=nbr_epoch,batch_size=batch_size,offset=offset, stacking=stacking)
		else :
			train_model(betavae,data_loader, optimizer, SAVE_PATH,path,nbr_epoch=nbr_epoch,batch_size=batch_size,offset=offset, stacking=stacking)
	else :
		if evaluate :
			evaluate_disentanglement(betavae, dataset, nbr_epoch=nbr_epoch)
		else :
			query_XYS(betavae, data_loader,path)



def train_model_head(betavae, gazehead, data_loader, optimizers, SAVE_PATH,path,nbr_epoch=100,batch_size=32, offset=0, stacking=False) :
	global use_cuda
	
	z_dim = betavae.z_dim
	img_depth=betavae.img_depth
	img_dim = betavae.img_dim

	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Debug :
	# fixed inputs for debugging
	fixed_z = Variable(torch.randn(45, z_dim))
	if use_cuda :
		fixed_z = fixed_z.cuda()

	sample = next(data_iter)
	fixed_x, _ = sample['image'], sample['landmarks']
	
	fixed_x = fixed_x.view( (-1, img_depth, img_dim, img_dim) )
	if not stacking :
		torchvision.utils.save_image(fixed_x.cpu(), './beta-data/{}/real_images.png'.format(path))
	else :
		fixed_x0 = fixed_x.view( (-1, 1, img_depth*img_dim, img_dim) )
		torchvision.utils.save_image(fixed_x0, './beta-data/{}/real_images.png'.format(path))


	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim)).float()
	if use_cuda :
		fixed_x = fixed_x.cuda()

	out = torch.zeros((1,1))

	# variations over the latent variable :
	sigma_mean = torch.ones((z_dim))
	mu_mean = torch.zeros((z_dim))

	best_loss = None
	best_model_wts = betavae.state_dict()
	
	cum_acc =0.0
	cum_merr = 0.0
	cum_stderr = 0.0
	

	for epoch in range(nbr_epoch):
		
		# Save generated variable images :
		nbr_steps = 8
		mu_mean /= batch_size
		sigma_mean /= batch_size
		gen_images = torch.ones( (nbr_steps, img_depth, img_dim, img_dim) )
		if stacking :
			gen_images = torch.ones( (nbr_steps, 1, img_depth*img_dim, img_dim) )
			
		for latent in range(z_dim) :
			#var_z0 = torch.stack( [mu_mean]*nbr_steps, dim=0)
			var_z0 = torch.zeros(nbr_steps, z_dim)
			val = mu_mean[latent]-sigma_mean[latent]
			step = 2.0*sigma_mean[latent]/nbr_steps
			print(latent,mu_mean[latent],step)
			for i in range(nbr_steps) :
				var_z0[i] = mu_mean
				var_z0[i][latent] = val
				val += step

			var_z0 = Variable(var_z0)
			if use_cuda :
				var_z0 = var_z0.cuda()


			gen_images_latent = betavae.decoder(var_z0)
			gen_images_latent = gen_images_latent.view(-1, img_depth, img_dim, img_dim).cpu().data
			if stacking :
				gen_images_latent = gen_images_latent.view( -1, 1, img_depth*img_dim, img_dim)
			gen_images = torch.cat( [gen_images,gen_images_latent], dim=0)

		#torchvision.utils.save_image(gen_images.data.cpu(),'./beta-data/{}/gen_images/dim{}/{}.png'.format(path,latent,(epoch+1)) )
		torchvision.utils.save_image(gen_images,'./beta-data/{}/gen_images/{}.png'.format(path,(epoch+offset+1)) )

		mu_mean = 0.0
		sigma_mean = 0.0

		epoch_loss = 0.0
		

		for i, sample in enumerate(data_loader):
			images = sample['image'].float()
			gaze = sample['landmarks'].float()

			# Save the reconstructed images
			if i % 100 == 0 :
				reconst_images, _, _ = betavae(fixed_x)
				reconst_images = reconst_images.view(-1, img_depth, img_dim, img_dim).cpu().data
				orimg = fixed_x.cpu().data.view(-1, img_depth, img_dim, img_dim)
				ri = torch.cat( [orimg, reconst_images], dim=2)
				if stacking :
					ri = reconst_images.view( (-1, 1, img_depth*img_dim, img_dim) )
				torchvision.utils.save_image(ri,'./beta-data/{}/reconst_images/{}.png'.format(path,(epoch+offset+1) ) )
				
			images = Variable( (images.view(-1, img_depth,img_dim, img_dim) ) )#.float()
			gaze = Variable( gaze. view((-1,2) ))

			if use_cuda :
				images = images.cuda() 
				gaze = gaze.cuda()

			out, mu, log_var = betavae(images)
			

			mu_mean += torch.mean(mu.data,dim=0)
			sigma_mean += torch.mean( torch.sqrt( torch.exp(log_var.data) ), dim=0 )

			# Compute :
			#reconstruction loss :
			reconst_loss = F.binary_cross_entropy( out, images, size_average=False)
			# expected log likelyhood :
			try :
				#expected_log_lik = torch.mean( Bernoulli( out.view((-1)) ).log_prob( images.view((-1)) ) )
				expected_log_lik = torch.mean( Bernoulli( out ).log_prob( images ) )
			except Exception as e :
				print(e)
				expected_log_lik = Variable(torch.ones(1).cuda())
			
			# kl divergence :
			kl_divergence = 0.5 * torch.mean( torch.sum( (mu**2 + torch.exp(log_var) - log_var -1), dim=1) )
			# ELBO :
			elbo = expected_log_lik - betavae.beta * kl_divergence
			

			#--------------------------------------------
			# MODEL 
			#--------------------------------------------
			# TOTAL LOSS :
			total_loss = reconst_loss + betavae.beta*kl_divergence
			# Backprop + Optimize :
			optimizers['model'].zero_grad()
			total_loss.backward(retain_graph=True)
			optimizers['model'].step()

			#--------------------------------------------
			#--------------------------------------------

			output_gaze = gazehead(mu)

			#--------------------------------------------
			# GazeHead : 
			#--------------------------------------------
			# TOTAL LOSS :
			gh_crit = nn.MSELoss()
			gh_total_loss = gh_crit(output_gaze,gaze) 
			# Backprop + Optimize :
			optimizers['head'].zero_grad()
			gh_total_loss.backward()
			optimizers['head'].step()
			
			#--------------------------------------------
			#--------------------------------------------

			# LOGS :
			error = output_gaze.cpu().data-gaze.cpu().data
			dist = torch.zeros( (error.size()[0],) )
			for j in range( dist.size()[0] ) :
				val = math.sqrt( error[j][0]**2 + error[j][1]**2 )
				dist[j] = val
			maxdist = math.sqrt( 0.35**2 + 0.35**2 )
			meanerror = (dist/(maxdist) ).mean()*100.0
			cum_merr = (cum_merr*i + meanerror)/(i+1)

			stderror = pow( ((dist/maxdist)*100.0-meanerror), 2.0).mean()
			cum_stderr = (cum_stderr + stderror)/(2)

			acc = (dist <= maxdist*0.1)
			acc = acc.numpy().mean()*100.0
			cum_acc = (cum_acc*i + acc)/(i+1)
			

			del images
			
			epoch_loss += gh_total_loss.cpu().data[0]

			if i % 10 == 0:
			    print ("Epoch[%d/%d], Step [%d/%d], VAE Total Loss: %.4f, "
			           "Reconst Loss: %.4f, KL Div: %.7f, E[ |~| p(x|theta)]: %.7f " 
			           %(epoch+1, nbr_epoch, i+1, iter_per_epoch, total_loss.data[0], 
			             reconst_loss.data[0], kl_divergence.data[0],expected_log_lik.exp().data[0]) )
			    print('Gaze:{:.4f} : Acc : {:.2f} % : Mean Error : {:.2f} % // Std error : {:.2f} %'.format( gh_total_loss.data[0], cum_acc,cum_merr,cum_stderr))
					

		if best_loss is None :
			#first validation : let us set the initialization but not save it :
			best_loss = epoch_loss
			model_best_model_wts = betavae.state_dict()
			gh_best_model_wts = gazehead.state_dict()
			# save best model weights :
			torch.save( model_best_model_wts, os.path.join(SAVE_PATH,'weights') )
			torch.save( gh_best_model_wts, gazehead.SAVE_PATH )
			print('Model VAE saved at : {}'.format(os.path.join(SAVE_PATH,'weights')) )
			print('Model GazeHead saved at : {}'.format( gazehead.SAVE_PATH ) )
		
		elif epoch_loss < best_loss:
			best_loss = epoch_loss
			model_best_model_wts = betavae.state_dict()
			gh_best_model_wts = gazehead.state_dict()
			# save best model weights :
			torch.save( model_best_model_wts, os.path.join(SAVE_PATH,'weights') )
			torch.save( gh_best_model_wts, gazehead.SAVE_PATH )
			print('Model VAE saved at : {}'.format(os.path.join(SAVE_PATH,'weights')) )
			print('Model GazeHead saved at : {}'.format( gazehead.SAVE_PATH ) )
		



def train_model(betavae,data_loader, optimizer, SAVE_PATH,path,nbr_epoch=100,batch_size=32, offset=0, stacking=False) :
	global use_cuda
	
	z_dim = betavae.z_dim
	img_depth=betavae.img_depth
	img_dim = betavae.img_dim

	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Debug :
	# fixed inputs for debugging
	fixed_z = Variable(torch.randn(45, z_dim))
	if use_cuda :
		fixed_z = fixed_z.cuda()

	sample = next(data_iter)
	fixed_x, _ = sample['image'], sample['landmarks']
	
	fixed_x = fixed_x.view( (-1, img_depth, img_dim, img_dim) )
	if not stacking :
		torchvision.utils.save_image(fixed_x.cpu(), './beta-data/{}/real_images.png'.format(path))
	else :
		fixed_x0 = fixed_x.view( (-1, 1, img_depth*img_dim, img_dim) )
		torchvision.utils.save_image(fixed_x0, './beta-data/{}/real_images.png'.format(path))


	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim)).float()
	if use_cuda :
		fixed_x = fixed_x.cuda()

	out = torch.zeros((1,1))

	# variations over the latent variable :
	sigma_mean = torch.ones((z_dim))
	mu_mean = torch.zeros((z_dim))

	best_loss = None
	best_model_wts = betavae.state_dict()
	
	for epoch in range(nbr_epoch):
		
		# Save generated variable images :
		nbr_steps = 8
		mu_mean /= batch_size
		sigma_mean /= batch_size
		gen_images = torch.ones( (nbr_steps, img_depth, img_dim, img_dim) )
		if stacking :
			gen_images = torch.ones( (nbr_steps, 1, img_depth*img_dim, img_dim) )
			
		for latent in range(z_dim) :
			#var_z0 = torch.stack( [mu_mean]*nbr_steps, dim=0)
			var_z0 = torch.zeros(nbr_steps, z_dim)
			val = mu_mean[latent]-sigma_mean[latent]
			step = 2.0*sigma_mean[latent]/nbr_steps
			print(latent,mu_mean[latent],step)
			for i in range(nbr_steps) :
				var_z0[i] = mu_mean
				var_z0[i][latent] = val
				val += step

			var_z0 = Variable(var_z0)
			if use_cuda :
				var_z0 = var_z0.cuda()


			gen_images_latent = betavae.decoder(var_z0)
			gen_images_latent = gen_images_latent.view(-1, img_depth, img_dim, img_dim).cpu().data
			if stacking :
				gen_images_latent = gen_images_latent.view( -1, 1, img_depth*img_dim, img_dim)
			gen_images = torch.cat( [gen_images,gen_images_latent], dim=0)

		#torchvision.utils.save_image(gen_images.data.cpu(),'./beta-data/{}/gen_images/dim{}/{}.png'.format(path,latent,(epoch+1)) )
		torchvision.utils.save_image(gen_images,'./beta-data/{}/gen_images/{}.png'.format(path,(epoch+offset+1)) )

		mu_mean = 0.0
		sigma_mean = 0.0

		epoch_loss = 0.0
		

		for i, sample in enumerate(data_loader):
			images = sample['image'].float()
			# Save the reconstructed images
			if i % 100 == 0 :
				reconst_images, _, _ = betavae(fixed_x)
				reconst_images = reconst_images.view(-1, img_depth, img_dim, img_dim).cpu().data
				orimg = fixed_x.cpu().data.view(-1, img_depth, img_dim, img_dim)
				ri = torch.cat( [orimg, reconst_images], dim=2)
				if stacking :
					ri = reconst_images.view( (-1, 1, img_depth*img_dim, img_dim) )
				torchvision.utils.save_image(ri,'./beta-data/{}/reconst_images/{}.png'.format(path,(epoch+offset+1) ) )
				
			images = Variable( (images.view(-1, img_depth,img_dim, img_dim) ) )#.float()
			
			if use_cuda :
				images = images.cuda() 

			out, mu, log_var = betavae(images)
			
			mu_mean += torch.mean(mu.data,dim=0)
			sigma_mean += torch.mean( torch.sqrt( torch.exp(log_var.data) ), dim=0 )

			# Compute :
			#reconstruction loss :
			reconst_loss = F.binary_cross_entropy( out, images, size_average=False)
			#reconst_loss = nn.MultiLabelSoftMarginLoss()(input=out_logits, target=images)
			#reconst_loss = F.binary_cross_entropy_with_logits( input=out, target=images, size_average=False)
			#reconst_loss = F.binary_cross_entropy( Bernoulli(out).sample(), images, size_average=False)
			#reconst_loss = torch.mean( (out.view(-1) - images.view(-1))**2 )
			
			# expected log likelyhood :
			try :
				#expected_log_lik = torch.mean( Bernoulli( out.view((-1)) ).log_prob( images.view((-1)) ) )
				expected_log_lik = torch.mean( Bernoulli( out ).log_prob( images ) )
			except Exception as e :
				print(e)
				expected_log_lik = Variable(torch.ones(1).cuda())
			
			# kl divergence :
			kl_divergence = 0.5 * torch.mean( torch.sum( (mu**2 + torch.exp(log_var) - log_var -1), dim=1) )
			#kl_divergence = 0.5 * torch.sum( (mu**2 + torch.exp(log_var) - log_var -1) )

			# ELBO :
			elbo = expected_log_lik - betavae.beta * kl_divergence
			
			# TOTAL LOSS :
			total_loss = reconst_loss + betavae.beta*kl_divergence
			#total_loss = reconst_loss
			#total_loss = -elbo

			# Backprop + Optimize :
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()

			del images
			
			epoch_loss += total_loss.cpu().data[0]

			if i % 10 == 0:
			    print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
			           "Reconst Loss: %.4f, KL Div: %.7f, E[ |~| p(x|theta)]: %.7f " 
			           %(epoch+1, nbr_epoch, i+1, iter_per_epoch, total_loss.data[0], 
			             reconst_loss.data[0], kl_divergence.data[0],expected_log_lik.exp().data[0]) )

		if best_loss is None :
			#first validation : let us set the initialization but not save it :
			best_loss = epoch_loss
			best_model_wts = betavae.state_dict()
			# save best model weights :
			torch.save( best_model_wts, os.path.join(SAVE_PATH,'weights') )
			print('Model saved at : {}'.format(os.path.join(SAVE_PATH,'weights')) )
		elif epoch_loss < best_loss:
			best_loss = epoch_loss
			best_model_wts = betavae.state_dict()
			# save best model weights :
			torch.save( best_model_wts, os.path.join(SAVE_PATH,'weights') )
			print('Model saved at : {}'.format(os.path.join(SAVE_PATH,'weights')) )


def query_XYS(betavae,data_loader,path):
	global use_cuda

	z_dim = betavae.z_dim
	img_depth=betavae.img_depth
	img_dim = betavae.img_dim
	
	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Debug :
	# fixed inputs for debugging
	fixed_z = Variable(torch.randn(45, z_dim))
	if use_cuda :
		fixed_z = fixed_z.cuda()

	sample = next(data_iter)
	fixed_x, _ = sample['image'], sample['landmarks']
		
	fixed_x = fixed_x.view( (-1, img_depth, img_dim, img_dim) )
	torchvision.utils.save_image(fixed_x.cpu(), './beta-data/{}/real_images_query.png'.format(path))
	
	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim)).float()
	if use_cuda :
		fixed_x = fixed_x.cuda()

	# variations over the latent variable :
	sigma_mean = 3.0*torch.ones((z_dim))
	mu_mean = torch.zeros((z_dim))

	# Save generated variable images :
	nbr_steps = 8
	gen_images = torch.ones( (8, img_depth, img_dim, img_dim) )

	for latent in range(z_dim) :
		#var_z0 = torch.stack( [mu_mean]*nbr_steps, dim=0)
		var_z0 = torch.zeros(nbr_steps, z_dim)
		val = mu_mean[latent]-sigma_mean[latent]
		step = 2.0*sigma_mean[latent]/nbr_steps
		print(latent,mu_mean[latent]-sigma_mean[latent],mu_mean[latent],mu_mean[latent]+sigma_mean[latent])
		for i in range(nbr_steps) :
			var_z0[i] = mu_mean
			var_z0[i][latent] = val
			val += step

		var_z0 = Variable(var_z0)
		if use_cuda :
			var_z0 = var_z0.cuda()


		gen_images_latent = betavae.decoder(var_z0)
		gen_images_latent = gen_images_latent.view(-1, img_depth, img_dim, img_dim).cpu().data
		gen_images = torch.cat( [gen_images,gen_images_latent], dim=0)

	#torchvision.utils.save_image(gen_images.data.cpu(),'./beta-data/{}/gen_images/dim{}/{}.png'.format(path,latent,(epoch+1)) )
	torchvision.utils.save_image(gen_images,'./beta-data/{}/gen_images/query.png'.format(path) )


	reconst_images, _, _ = betavae(fixed_x)
	reconst_images = reconst_images.view(-1, img_depth, img_dim, img_dim).cpu().data
	orimg = fixed_x.cpu().data.view(-1, img_depth, img_dim, img_dim)
	ri = torch.cat( [orimg, reconst_images], dim=2)
	torchvision.utils.save_image(ri,'./beta-data/{}/reconst_images/query.png'.format(path ) )
	

def generateTarget(latent_dim=3,idx_latent=0, batch_size=8 ) :
	target = torch.zeros( (1, latent_dim))
	target[0,idx_latent] = 1.0
	target = torch.cat( batch_size*[target], dim=0)

	target = Variable(target)
	global use_cuda 
	if use_cuda :
		target = target.cuda()

	return target

def generatePairs(index) :
	nbrel = len(index)
	
	# if not even :
	if nbrel % 2 :
		nbrel -= 1
		print('NBR ELEMENT : {}'.format(nbrel) )
		# ditch the last one...

	id1 = []
	id2 = []

	for i in range( 0, nbrel, 2 ) :
		id1.append( index[i] )
		id2.append( index[i+1] )

	return id1, id2

def evaluate_disentanglement(model,dataset,nbr_epoch=20) :
	from datasetXYS import generateIDX, generateClassifier

	global use_cuda
	lr = 1e-4
	
	indexes = generateIDX(dataset)
	nbr_latent = len(indexes)
	disentanglement_measure = [0.0] * nbr_latent 

	classifier = generateClassifier(input_dim=model.z_dim, output_dim=nbr_latent)
	if use_cuda :
		classifier = classifier.cuda()

	optimizer = torch.optim.Adam( classifier.parameters(), lr=lr)

	batch_size = 1
	iteration_training = 0
	cum_training_acc = 0.0

	for phase in ['train','val'] :
		if phase == 'val' :
			nbrepoch = 1
		else :
			nbrepoch = nbr_epoch

		for epoch in range(nbrepoch) :
			cum_epoch_acc = 0.0
			iteration_epoch = 0

			for idx_latent in range(3) :
				index = indexes[idx_latent]
				
				nbr_classes = len(index)
				iteration_latent = 0
				cum_latent_acc = 0.0

				for cl in range(nbr_classes) :
					#print('CLASS : {} : nbr element = {}'.format(cl,len(index[cl]) ) )
					index1, index2 = generatePairs(index[cl])
					nbrel = len(index1)
					
					print('')
					#print('IDX LATENT : {} // NBR ELEMENT : {}'.format(idx_latent,nbrel))				
					
					for it in range(nbrel) :
						sample1 = dataset[index1[it]]
						img1 = sample1['image'].unsqueeze(0)
						sample2 = dataset[index2[it]]
						img2 = sample2['image'].unsqueeze(0)


						#img1 = Variable( (img1.view(-1, model.img_depth, model.img_dim, model.img_dim) ) ).float()
						#img2 = Variable( (img2.view(-1, model.img_depth, model.img_dim, model.img_dim) ) ).float()
						img1 = Variable( img1 ).float()
						img2 = Variable( img2 ).float()
						
						if use_cuda :
							img1 = img1.cuda() 
							img2 = img2.cuda() 


						_, mu1, log_var1 = model(img1)
						_, mu2, log_var2 = model(img2)
						
						z_diff = torch.abs(mu2-mu1)
						#av_z_diff = z_diff/float(nbrel)

						target = generateTarget(latent_dim=3, idx_latent=idx_latent, batch_size=1)

						#logits = classifier(av_z_diff)
						logits = classifier(z_diff)


						# Accuracy :
						acc = (logits.cpu().data.max(1)[1] == idx_latent)
						acc = acc.numpy().mean()*100.0
						cum_latent_acc = (cum_latent_acc*iteration_latent + acc)/(iteration_latent+1)
						iteration_latent += 1
						cum_epoch_acc = (cum_epoch_acc*iteration_epoch + acc)/(iteration_epoch+1)
						iteration_epoch += 1
						cum_training_acc = (cum_training_acc*iteration_training + acc)/(iteration_training+1)
						iteration_training += 1

						print('{} EPOCH : {} :: iteration {}/{} :: Cumulative Accuracy : {} // Cumulative Latent {} Accuracy : {}'.format(phase, epoch, it, nbrel, cum_epoch_acc, idx_latent, cum_latent_acc), end='\r')


						# Loss :
						loss = F.binary_cross_entropy( logits, target)
						
						if phase == 'train' :
							# Training :
							loss.backward()
							
							if iteration_latent % batch_size == 0 :
								optimizer.step()
								optimizer.zero_grad()

			print('')
			print('-'*20)
			print('{} EPOCH : {}/{} :: Cumulative Accuracy : {} // Cumulative Latent {} Accuracy : {}'.format(phase, epoch, nbrepoch, cum_epoch_acc, idx_latent, cum_latent_acc))
			print('-'*20)
		


if __name__ == '__main__' :
	import argparse
	parser = argparse.ArgumentParser(description='beta-VAE')
	parser.add_argument('--train',action='store_true',default=False)
	parser.add_argument('--query',action='store_true',default=False)
	parser.add_argument('--evaluate',action='store_true',default=False)
	parser.add_argument('--stacked',action='store_true',default=False)
	parser.add_argument('--train_head',action='store_true',default=False)
	parser.add_argument('--offset', type=int, default=0)
	parser.add_argument('--batch', type=int, default=32)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--latent', type=int, default=3)
	parser.add_argument('--lr', type=float, default=1e-4)
	args = parser.parse_args()

	if args.train :
		setting(offset=args.offset,batch_size=args.batch,train=True,nbr_epoch=args.epoch,stacking=args.stacked,lr=args.lr,z_dim=args.latent)
	elif args.train_head :
		setting(offset=args.offset,batch_size=args.batch,train=True,nbr_epoch=args.epoch,stacking=args.stacked,lr=args.lr,z_dim=args.latent, train_head=True)

	if args.query :
		setting(train=False,stacking=args.stacked,lr=args.lr,z_dim=args.latent)

	if args.evaluate :
		setting(train=False,evaluate=True,nbr_epoch=args.epoch,stacking=args.stacked,lr=args.lr,z_dim=args.latent)