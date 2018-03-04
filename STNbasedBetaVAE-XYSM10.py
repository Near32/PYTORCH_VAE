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
from math import floor
from PIL import Image


from models import Rescale, betaVAE, betaVAEdSprite, betaVAEXYS, betaVAEXYS2, betaVAEXYS3, STNbasedBetaVAEXYS3, Bernoulli, GazeHead
from datasetXYS import load_dataset_XYS, load_dataset_XYSM10

use_cuda = True


def setting(args,nbr_epoch=100,offset=0,train=True,batch_size=32, evaluate=False,stacking=False,lr = 1e-5,z_dim = 3, beta=1.0, train_head=False, data='XYS'):	
	size = 256
	
	if 'XYSM10' in data :
		dataset = load_dataset_XYSM10(img_dim=size,stacking=False) 
	else :
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
	beta = beta
	betavae = betaVAEXYS2(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	'''

	frompath = True
	img_dim = size
	img_depth=3
	conv_dim = 8#32
	global use_cuda
	net_depth = 6
	beta = beta
	#betavae = betaVAEXYS3(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	betavae = STNbasedBetaVAEXYS3(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	print(betavae)
		
	# LOADING :
	path = '{}--STN--img{}-lr{}-beta{}-layers{}-z{}-conv{}'.format(args.data,img_dim,lr,beta,net_depth,z_dim,conv_dim)
	
	if not os.path.exists( './beta-data/{}/'.format(path) ) :
		os.mkdir('./beta-data/{}/'.format(path))
	if not os.path.exists( './beta-data/{}/gen_images/'.format(path) ) :
			os.mkdir('./beta-data/{}/gen_images/'.format(path))
	if not os.path.exists( './beta-data/{}/reconst_images/'.format(path) ) :
			os.mkdir('./beta-data/{}/reconst_images/'.format(path))
	
	
	SAVE_PATH = './beta-data/{}'.format(path) 
	path1 = os.path.join(SAVE_PATH,'weights')
	path2 = os.path.join(SAVE_PATH,'temp.weights')
	if args.querySTN or args.query :
		path2 = os.path.join(SAVE_PATH,'weights')
		path1 = os.path.join(SAVE_PATH,'temp.weights')
		
	if frompath :
		try :
			betavae.load_state_dict( torch.load( path1 ) )
			print('NET LOADING : from {} : OK.'.format(path1) )
		except Exception as e :
			print('EXCEPTION : NET LOADING : {}'.format(e) )
			try :
				betavae.load_state_dict( torch.load( path2) )
				print('NET LOADING : from {} : OK.'.format(path2))
			except Exception as e :
				print('EXCEPTION : NET LOADING : {}'.format(e) )

	# GAZE HEAD :
	if train_head :
		gazehead = GazeHead(outdim=2, nbr_latents=z_dim, use_cuda=use_cuda)

		# LOADING :
		gh_path = path
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
			if args.querySTN :
				query_STN(betavae, data_loader,path)
			else :
				query_XYS(betavae, data_loader,path,args)



def train_model_head(betavae, gazehead, data_loader, optimizers, SAVE_PATH,path,nbr_epoch=100,batch_size=32, offset=0, stacking=False) :
	global use_cuda
	
	train_model_too = False

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

			if train_model_too :
				out, mu, log_var = betavae(images)
			else :
				out = images
				z, mu, log_var = betavae.encode(images)
			

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
			
			output_gaze = gazehead(mu)
			gh_crit = nn.MSELoss()
			
			#--------------------------------------------
			# TOTAL LOSS :
			gh_total_loss = gh_crit(output_gaze,gaze) 
			total_loss = reconst_loss + betavae.beta*kl_divergence + gh_total_loss			
			#--------------------------------------------
			#--------------------------------------------
			# Backprop + Optimize :
			optimizers['head'].zero_grad()
			optimizers['model'].zero_grad()
			
			if train_model_too :
				total_loss.backward(retain_graph=True)
				optimizers['model'].step()

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
		nbr_steps = args.querySTEPS
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
				
				model_wts = betavae.state_dict()
				torch.save( model_wts, os.path.join(SAVE_PATH,'temp.weights') )
				print('Model saved at : {}'.format(os.path.join(SAVE_PATH,'temp.weights')) )

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

			#--------------------------
			#betavae.encoder.localization.zero_grad()
			#betavae.encoder.fc_loc.zero_grad()
			nn.utils.clip_grad_norm( betavae.encoder.localization.parameters(), args.clip)
			nn.utils.clip_grad_norm( betavae.encoder.fc_loc.parameters(), args.clip)
			#--------------------------
			
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


def query_XYS(betavae,data_loader,path,args):
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
	sigma_mean = args.queryVAR*torch.ones((z_dim))
	mu_mean = torch.zeros((z_dim))

	# Save generated variable images :
	nbr_steps = 16
	gen_images = torch.ones( (nbr_steps, img_depth, img_dim, img_dim) )

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


def query_STN(betavae,data_loader,path):
	global use_cuda

	z_dim = betavae.z_dim
	img_depth=betavae.img_depth
	img_dim = betavae.img_dim
	
	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Debug :
	# fixed inputs for debugging
	sample = next(data_iter)
	fixed_x, _ = sample['image'], sample['landmarks']
	batch_size = fixed_x.size()[0]

	fixed_x = fixed_x.view( (-1, img_depth, img_dim, img_dim) )
	torchvision.utils.save_image(fixed_x.cpu(), './beta-data/{}/real_images_querySTN.png'.format(path))
	
	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim)).float()
	if use_cuda :
		fixed_x = fixed_x.cuda()

	stn_output = betavae.encoder.stn(fixed_x)
	stn_output = stn_output.view(-1, img_depth, img_dim, img_dim).cpu().data
	#stn_output = stn_output.view(batch_size, img_depth, -1, img_dim).cpu().data
	#orimg = fixed_x.cpu().data.view(-1, img_depth, img_dim, img_dim)
	#ri = torch.cat( [orimg, stn_output], dim=2)
	#torchvision.utils.save_image(ri,'./beta-data/{}/reconst_images/querySTN.png'.format(path ) )
	torchvision.utils.save_image(stn_output,'./beta-data/{}/reconst_images/querySTN.png'.format(path ) )
	print('STN sample saved at : ./beta-data/{}/reconst_images/querySTN.png'.format(path ) )


def generateIndex4El(cl_indexes,batch_size) :
	nbr_class = len(cl_indexes)
	nbr_el_per_class = [0]*nbr_class
	nbr_iter_per_class = [0]*nbr_class
	index4el = list()
	
	for cl,el_cl_indexes in enumerate(cl_indexes) :
		nbr_el_per_class[cl] = len(el_cl_indexes)
		nbr_iter_per_class[cl] = floor(nbr_el_per_class[cl]/batch_size)
		for itera in range(nbr_iter_per_class[cl]) :
			index4el.append(cl)
	
	return index4el


def evaluate_disentanglement(model,dataset,nbr_epoch=20) :
	from datasetXYS import generateIDX, generateClassifier
	model.eval()

	global use_cuda
	lr = 1e-2
	
	indexes = generateIDX(dataset)
	nbr_latent = len(indexes)
	disentanglement_measure = [0.0] * nbr_latent 

	classifiers = []
	optimizers = []
	for i in range(nbr_latent) :
		classifiers.append(  generateClassifier(input_dim=model.z_dim, output_dim=nbr_latent) )
		if use_cuda :
			classifiers[-1] = classifiers[-1].cuda()

		optimizers.append( torch.optim.Adagrad( classifiers[-1].parameters(), lr=lr) )

	batch_size = 2*10
	# Accumulator for the z latent representations of each sample of the dataset :
	index2z = [None]*len(dataset)

	for phase in ['train','val'] :
		if phase == 'val' :
			nbrepoch = 1
		else :
			nbrepoch = nbr_epoch

		for epoch in range(nbrepoch) :

			for idx_latent in range(nbr_latent) :
				cum_latent_acc = 0.0
				iteration_latent = 0
				# List of list of elements from the same classes :
				cl_indexes = indexes[idx_latent]
				# List of class indexes for every batch of element of the latent :
				cl_index4el = generateIndex4El(cl_indexes,batch_size)
				# Shuffled dataset of all the classes that needs to be samples in order to go through the whole dataset :
				cl_dataloader = torch.utils.data.DataLoader(dataset=cl_index4el,batch_size=1, shuffle=True)
				
				for i,cl_sample in enumerate(cl_dataloader) :
					cl_sample = int(cl_sample[0])
					# List of indexes of all the elements of the same sampled class :
					el_cl_index = cl_indexes[cl_sample]
					# Shuffled dataset of all the element's indexes that have the same sampled class :
					el_cl_dataloader = torch.utils.data.DataLoader(dataset=el_cl_index,batch_size=batch_size, shuffle=True)
					
					for j,el_index_samples in enumerate(el_cl_dataloader) :
						# Sampling a batch of indexes of element that shares the same class.
						# Sampling, encoding and summing the elements :
						z_diff_avg = 0
						it = 0
						for index in el_index_samples :
							
							it+=1
							if index2z[index] is None :
								x_sample = dataset[index]
								img = x_sample['image'].unsqueeze(0)
								img = Variable( img ).float()
								
								if use_cuda :
									img = img.cuda() 
								
								_, z, _ = model.encode(img1)
								index2z[index] = z
							else :
								z = index2z[index]

							# let us 'diff_sum' :
							z_diff_avg += pow(-1,it)*z

						z_diff_avg /= batch_size/2

						# Forward pass :
						logits = classifiers[idx_latent](z_diff_avg)

						# Loss :
						target = generateTarget(latent_dim=3, idx_latent=idx_latent, batch_size=1)
						loss = F.binary_cross_entropy( logits, target)

						# Accuracy :
						acc = (logits.cpu().data.max(1)[1] == idx_latent)
						acc = acc.numpy().mean()*100.0
						cum_latent_acc = (cum_latent_acc*iteration_latent + acc)/(iteration_latent+1)
						iteration_latent += 1
						
						print('{} EPOCH : {} :: iteration {}/{} :: Latent {} Accuracy : {}'.format(phase, epoch, iteration, nbr_iterations, idx_latent, cum_latent_acc), end='\r')

						# Training :
						if phase == 'train' :
							loss.backward()
							optimizer.step()
							optimizer.zero_grad()

				print('')
				print('-'*20)
				print('{} EPOCH : {} :: iteration {}/{} :: Latent {} Accuracy : {}'.format(phase, epoch, iteration, nbr_iterations, idx_latent, cum_latent_acc), end='\r')
				print('-'*20)
			




if __name__ == '__main__' :
	import argparse
	parser = argparse.ArgumentParser(description='beta-VAE')
	parser.add_argument('--train',action='store_true',default=False)
	parser.add_argument('--query',action='store_true',default=False)
	parser.add_argument('--querySTN',action='store_true',default=False)
	parser.add_argument('--evaluate',action='store_true',default=False)
	parser.add_argument('--train_head',action='store_true',default=False)
	parser.add_argument('--offset', type=int, default=0)
	parser.add_argument('--batch', type=int, default=32)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--latent', type=int, default=3)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--beta', type=float, default=1e0)
	parser.add_argument('--data', type=str, default='XYS')
	parser.add_argument('--queryVAR', type=float, default=3.0)
	parser.add_argument('--querySTEPS', type=int, default=8)
	parser.add_argument('--clip', type=float, default=1e-5)
	args = parser.parse_args()

	print(args)

	if args.train :
		setting(args,offset=args.offset,batch_size=args.batch,train=True,nbr_epoch=args.epoch,lr=args.lr,z_dim=args.latent, beta=args.beta, data=args.data)
	elif args.train_head :
		setting(args,offset=args.offset,batch_size=args.batch,train=True,nbr_epoch=args.epoch,lr=args.lr,z_dim=args.latent, beta=args.beta, train_head=True, data=args.data)

	if args.query :
		setting(args,train=False,lr=args.lr,z_dim=args.latent, beta=args.beta, data=args.data)
	elif args.querySTN :
		setting(args,train=False,lr=args.lr,z_dim=args.latent, beta=args.beta, data=args.data)

	if args.evaluate :
		setting(args,train=False,evaluate=True,nbr_epoch=args.epoch,lr=args.lr,z_dim=args.latent, beta=args.beta)