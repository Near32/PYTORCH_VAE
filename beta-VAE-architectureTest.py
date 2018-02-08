import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from skimage import io, transform
import numpy as np

from PIL import Image


from models import Rescale, betaVAE, betaVAEdSprite, betaVAEXYS, Bernoulli
from datasetXYS import load_dataset_XYS

def test_mnist():
	import os
	import torchvision
	from torchvision import datasets, transforms

	size = 64
	batch_size = 128
	dataset = datasets.MNIST(root='./data',
		                     train=True,
		                     #transform=transforms.ToTensor(),
							 transform=transforms.Compose([
	                                               Rescale( (size,size) ),
	                                               transforms.ToTensor()]),
	                         download=True) 


	# Data loader
	data_loader = torch.utils.data.DataLoader(dataset=dataset,
    	                                      batch_size=batch_size, 
        	                                  shuffle=True)
	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Model :
	z_dim = 12
	img_dim = size
	img_depth=1
	conv_dim = 32
	use_cuda = True#False
	net_depth = 3
	beta = 5e0
	betavae = betaVAE(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	print(betavae)


	# Optim :
	lr = 1e-4
	optimizer = torch.optim.Adam( betavae.parameters(), lr=lr)

	# Debug :
	# fixed inputs for debugging
	fixed_z = Variable(torch.randn(100, z_dim))
	if use_cuda :
		fixed_z = fixed_z.cuda()

	fixed_x, _ = next(data_iter)
	

	path = 'test--mnist-beta{}-layers{}-z{}-conv{}-lr{}'.format(beta,net_depth,z_dim,conv_dim,lr)
	if not os.path.exists( './beta-data/{}/'.format(path) ) :
		os.mkdir('./beta-data/{}/'.format(path))
	if not os.path.exists( './beta-data/{}/gen_images/'.format(path) ) :
			os.mkdir('./beta-data/{}/gen_images/'.format(path))
	
	
	fixed_x = fixed_x.view( (-1, img_depth, img_dim, img_dim) )
	torchvision.utils.save_image(fixed_x.cpu(), './beta-data/{}/real_images.png'.format(path))
	
	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim))
	if use_cuda :
		fixed_x = fixed_x.cuda()

	out = torch.zeros((1,1))

	# variations over the latent variable :
	sigma_mean = torch.ones((z_dim))
	mu_mean = torch.zeros((z_dim))

	for epoch in range(50):
		# Save the reconstructed images
		reconst_images, _, _ = betavae(fixed_x)
		reconst_images = reconst_images.view(-1, img_depth, img_dim, img_dim)
		torchvision.utils.save_image(reconst_images.data.cpu(),'./beta-data/{}/reconst_images_{}.png'.format(path,(epoch+1)) )

		# Save generated variable images :
		nbr_steps = 8
		mu_mean /= batch_size
		sigma_mean /= batch_size
		gen_images = torch.ones( (8, img_depth, img_dim, img_dim) )

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
			gen_images = torch.cat( [gen_images,gen_images_latent], dim=0)

		#torchvision.utils.save_image(gen_images.data.cpu(),'./beta-data/{}/gen_images/dim{}/{}.png'.format(path,latent,(epoch+1)) )
		torchvision.utils.save_image(gen_images,'./beta-data/{}/gen_images/{}.png'.format(path,(epoch+1)) )

		mu_mean = 0.0
		sigma_mean = 0.0

		for i, (images, _) in enumerate(data_loader):

			images = Variable( (images.view(-1,1,img_dim, img_dim) ) )
			if use_cuda :
				images = images.cuda() 

			out, mu, log_var = betavae(images)

			mu_mean += torch.mean(mu.data,dim=0)
			sigma_mean += torch.mean( torch.sqrt( torch.exp(log_var.data) ), dim=0 )

			# Compute :
			#reconstruction loss :
			reconst_loss = F.binary_cross_entropy(out, images, size_average=False)
			#reconst_loss = torch.mean( (out.view(-1) - images.view(-1))**2 )

			# expected log likelyhood :
			expected_log_lik = torch.mean( Bernoulli( out.view((-1)) ).log_prob( images.view((-1)) ) )
			#expected_log_lik = torch.mean( Bernoulli( out ).log_prob( images ) )

			# kl divergence :
			#kl_divergence = 0.5 * torch.mean( torch.sum( (mu**2 + torch.exp(log_var) - log_var -1), dim=1) )
			kl_divergence = 0.5 * torch.sum( (mu**2 + torch.exp(log_var) - log_var -1) )

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

			if i % 100 == 0:
			    print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
			           "Reconst Loss: %.4f, KL Div: %.7f, E[ |~| p(x|theta)]: %.7f " 
			           %(epoch+1, 50, i+1, iter_per_epoch, total_loss.data[0], 
			             reconst_loss.data[0], kl_divergence.data[0],expected_log_lik.exp().data[0]) )





def test_dSprite():
	import os
	import matplotlib.pyplot as plt
	import torchvision
	from torchvision import datasets, transforms
	from datasets import dSpriteDataset

	size = 64
	batch_size = 256
	root = './dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
	dataset = dSpriteDataset(root=root,
							 transform=transforms.Compose([
	                                               Rescale( (size,size) ),
	                                               transforms.ToTensor()])
	                        ) 

	# Data loader
	data_loader = torch.utils.data.DataLoader(dataset=dataset,
    	                                      batch_size=batch_size, 
        	                                  shuffle=True)
	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Model :
	frompath = True
	'''
	z_dim = 10
	img_dim = size
	img_depth=1
	conv_dim = 64
	use_cuda = True#False
	net_depth = 3
	beta = 1e0
	betavae = betaVAEdSprite(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	'''
	# Model :
	z_dim = 10
	img_dim = size
	img_depth=1
	conv_dim = 32
	use_cuda = True#False
	net_depth = 3
	beta = 100e0
	betavae = betaVAE(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	print(betavae)


	# Optim :
	lr = 1e-5
	optimizer = torch.optim.Adam( betavae.parameters(), lr=lr)
	#optimizer = torch.optim.Adagrad( betavae.parameters(), lr=lr)
	#lr = 1e-3
	
	# Debug :
	# fixed inputs for debugging
	fixed_z = Variable(torch.randn(45, z_dim))
	if use_cuda :
		fixed_z = fixed_z.cuda()

	fixed_x, _ = next(data_iter)
	

	#path = 'dSprite--beta{}-layers{}-z{}-conv{}-lr{}'.format(beta,net_depth,z_dim,conv_dim,lr)
	path = 'testAblation--dSprite--beta{}-layers{}-z{}-conv{}'.format(beta,net_depth,z_dim,conv_dim)
	if not os.path.exists( './beta-data/{}/'.format(path) ) :
		os.mkdir('./beta-data/{}/'.format(path))
	if not os.path.exists( './beta-data/{}/gen_images/'.format(path) ) :
			os.mkdir('./beta-data/{}/gen_images/'.format(path))
	if not os.path.exists( './beta-data/{}/reconst_images/'.format(path) ) :
			os.mkdir('./beta-data/{}/reconst_images/'.format(path))
	
	
	fixed_x = fixed_x.view( (-1, img_depth, img_dim, img_dim) )
	torchvision.utils.save_image(255*fixed_x.cpu(), './beta-data/{}/real_images.png'.format(path))
	
	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim))
	if use_cuda :
		fixed_x = fixed_x.cuda()

	out = torch.zeros((1,1))

	# variations over the latent variable :
	sigma_mean = torch.ones((z_dim))
	mu_mean = torch.zeros((z_dim))

	best_loss = None
	best_model_wts = betavae.state_dict()
	SAVE_PATH = './beta-data/{}'.format(path) 

	if frompath :
		try :
			betavae.load_state_dict( torch.load( os.path.join(SAVE_PATH,'weights')) )
			print('NET LOADING : OK.')
		except Exception as e :
			print('EXCEPTION : NET LOADING : {}'.format(e) )


	for epoch in range(50):
		
		# Save generated variable images :
		nbr_steps = 8
		mu_mean /= batch_size
		sigma_mean /= batch_size
		gen_images = torch.ones( (8, img_depth, img_dim, img_dim) )

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
			gen_images = torch.cat( [gen_images,gen_images_latent], dim=0)

		#torchvision.utils.save_image(gen_images.data.cpu(),'./beta-data/{}/gen_images/dim{}/{}.png'.format(path,latent,(epoch+1)) )
		torchvision.utils.save_image(255.0*gen_images,'./beta-data/{}/gen_images/{}.png'.format(path,(epoch+67)) )

		mu_mean = 0.0
		sigma_mean = 0.0

		epoch_loss = 0.0
		

		for i, (images, _) in enumerate(data_loader):
			# Save the reconstructed images
			if i % 100 == 0 :
				reconst_images, _, _ = betavae(fixed_x)
				reconst_images = reconst_images.view(-1, img_depth, img_dim, img_dim).cpu().data
				orimg = fixed_x.cpu().data.view(-1, img_depth, img_dim, img_dim)
				ri = torch.cat( [orimg, reconst_images], dim=2)
				torchvision.utils.save_image(255*ri,'./beta-data/{}/reconst_images/{}.png'.format(path,(epoch+67)) )
			
			
			images = Variable( (images.view(-1,1,img_dim, img_dim) ) )
			
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
			expected_log_lik = torch.mean( Bernoulli( out.view((-1)) ).log_prob( images.view((-1)) ) )
			#expected_log_lik = torch.mean( Bernoulli( out ).log_prob( images ) )

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

			if i % 100 == 0:
			    print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
			           "Reconst Loss: %.4f, KL Div: %.7f, E[ |~| p(x|theta)]: %.7f " 
			           %(epoch+1, 50, i+1, iter_per_epoch, total_loss.data[0], 
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


def test_XYS(offset=0):
	import os
	import matplotlib.pyplot as plt
	import torchvision
	from torchvision import datasets, transforms 
	from models import Bernoulli
	
	size = 256
	batch_size = 16#32
	
	dataset = load_dataset_XYS(img_dim=size)

	# Data loader
	data_loader = torch.utils.data.DataLoader(dataset=dataset,
    	                                      batch_size=batch_size, 
        	                                  shuffle=True)
	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Model :
	frompath = True
	'''
	z_dim = 10
	img_dim = size
	img_depth=1
	conv_dim = 64
	use_cuda = True#False
	net_depth = 3
	beta = 1e0
	betavae = betaVAEdSprite(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	'''
	# Model :
	z_dim = 4
	img_dim = size
	img_depth=3
	conv_dim = 32
	use_cuda = True#False
	net_depth = 5
	beta = 5000e0
	betavae = betaVAEXYS(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	print(betavae)


	# Optim :
	lr = 1e-5
	optimizer = torch.optim.Adam( betavae.parameters(), lr=lr)
	#optimizer = torch.optim.Adagrad( betavae.parameters(), lr=lr)
	#lr = 1e-3
	
	# Debug :
	# fixed inputs for debugging
	fixed_z = Variable(torch.randn(45, z_dim))
	if use_cuda :
		fixed_z = fixed_z.cuda()

	sample = next(data_iter)
	fixed_x, _ = sample['image'], sample['landmarks']
		

	path = 'test--XYS--img{}-lr{}-beta{}-layers{}-z{}-conv{}'.format(img_dim,lr,beta,net_depth,z_dim,conv_dim)
	if not os.path.exists( './beta-data/{}/'.format(path) ) :
		os.mkdir('./beta-data/{}/'.format(path))
	if not os.path.exists( './beta-data/{}/gen_images/'.format(path) ) :
			os.mkdir('./beta-data/{}/gen_images/'.format(path))
	if not os.path.exists( './beta-data/{}/reconst_images/'.format(path) ) :
			os.mkdir('./beta-data/{}/reconst_images/'.format(path))
	
	
	fixed_x = fixed_x.view( (-1, img_depth, img_dim, img_dim) )
	torchvision.utils.save_image(fixed_x.cpu(), './beta-data/{}/real_images.png'.format(path))
	
	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim)).float()
	if use_cuda :
		fixed_x = fixed_x.cuda()

	out = torch.zeros((1,1))

	# variations over the latent variable :
	sigma_mean = torch.ones((z_dim))
	mu_mean = torch.zeros((z_dim))

	best_loss = None
	best_model_wts = betavae.state_dict()
	SAVE_PATH = './beta-data/{}'.format(path) 

	if frompath :
		try :
			betavae.load_state_dict( torch.load( os.path.join(SAVE_PATH,'weights')) )
			print('NET LOADING : OK.')
		except Exception as e :
			print('EXCEPTION : NET LOADING : {}'.format(e) )


	for epoch in range(50):
		
		# Save generated variable images :
		nbr_steps = 8
		mu_mean /= batch_size
		sigma_mean /= batch_size
		gen_images = torch.ones( (8, img_depth, img_dim, img_dim) )

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
			gen_images = torch.cat( [gen_images,gen_images_latent], dim=0)

		#torchvision.utils.save_image(gen_images.data.cpu(),'./beta-data/{}/gen_images/dim{}/{}.png'.format(path,latent,(epoch+1)) )
		torchvision.utils.save_image(gen_images,'./beta-data/{}/gen_images/{}.png'.format(path,(epoch+offset+1)) )

		mu_mean = 0.0
		sigma_mean = 0.0

		epoch_loss = 0.0
		

		for i, sample in enumerate(data_loader):
			images = sample['image']
			# Save the reconstructed images
			if i % 100 == 0 :
				reconst_images, _, _ = betavae(fixed_x)
				reconst_images = reconst_images.view(-1, img_depth, img_dim, img_dim).cpu().data
				orimg = fixed_x.cpu().data.view(-1, img_depth, img_dim, img_dim)
				ri = torch.cat( [orimg, reconst_images], dim=2)
				torchvision.utils.save_image(ri,'./beta-data/{}/reconst_images/{}.png'.format(path,(epoch+offset+1) ) )
			
			
			images = Variable( (images.view(-1, img_depth,img_dim, img_dim) ) ).float()
			
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
			expected_log_lik = torch.mean( Bernoulli( out.view((-1)) ).log_prob( images.view((-1)) ) )
			#expected_log_lik = torch.mean( Bernoulli( out ).log_prob( images ) )

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

			if i % 100 == 0:
			    print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
			           "Reconst Loss: %.4f, KL Div: %.7f, E[ |~| p(x|theta)]: %.7f " 
			           %(epoch+1, 50, i+1, iter_per_epoch, total_loss.data[0], 
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


def queryXYS():
	import os
	import matplotlib.pyplot as plt
	import torchvision
	from torchvision import datasets, transforms 
	from models import Bernoulli
	
	size = 256
	batch_size = 16#32
	
	dataset = load_dataset_XYS(img_dim=size)

	# Data loader
	data_loader = torch.utils.data.DataLoader(dataset=dataset,
    	                                      batch_size=batch_size, 
        	                                  shuffle=True)
	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Model :
	frompath = True
	'''
	z_dim = 10
	img_dim = size
	img_depth=1
	conv_dim = 64
	use_cuda = True#False
	net_depth = 3
	beta = 1e0
	betavae = betaVAEdSprite(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	'''
	# Model :
	z_dim = 4#10
	img_dim = size
	img_depth=3
	conv_dim = 32
	use_cuda = True#False
	net_depth = 5
	beta = 5000e0
	betavae = betaVAEXYS(beta=beta,net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	print(betavae)

	lr=1e-5

	# Debug :
	# fixed inputs for debugging
	fixed_z = Variable(torch.randn(45, z_dim))
	if use_cuda :
		fixed_z = fixed_z.cuda()

	sample = next(data_iter)
	fixed_x, _ = sample['image'], sample['landmarks']
		

	path = 'test--XYS--img{}-lr{}-beta{}-layers{}-z{}-conv{}'.format(img_dim,lr,beta,net_depth,z_dim,conv_dim)
	if not os.path.exists( './beta-data/{}/'.format(path) ) :
		os.mkdir('./beta-data/{}/'.format(path))
	if not os.path.exists( './beta-data/{}/gen_images/'.format(path) ) :
			os.mkdir('./beta-data/{}/gen_images/'.format(path))
	if not os.path.exists( './beta-data/{}/reconst_images/'.format(path) ) :
			os.mkdir('./beta-data/{}/reconst_images/'.format(path))
	
	
	fixed_x = fixed_x.view( (-1, img_depth, img_dim, img_dim) )
	torchvision.utils.save_image(fixed_x.cpu(), './beta-data/{}/real_images_query.png'.format(path))
	
	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim)).float()
	if use_cuda :
		fixed_x = fixed_x.cuda()

	out = torch.zeros((1,1))

	# variations over the latent variable :
	sigma_mean = 3.0*torch.ones((z_dim))
	mu_mean = torch.zeros((z_dim))

	SAVE_PATH = './beta-data/{}'.format(path) 

	if frompath :
		try :
			betavae.load_state_dict( torch.load( os.path.join(SAVE_PATH,'weights')) )
			print('NET LOADING : OK.')
		except Exception as e :
			print('EXCEPTION : NET LOADING : {}'.format(e) )


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
	

if __name__ == '__main__' :
	import argparse
	parser = argparse.ArgumentParser(description='beta-VAE')
	parser.add_argument('--train',action='store_true',default=False)
	parser.add_argument('--offset', type=int, default=0)
	args = parser.parse_args()

	if args.train :
		#test_mnist()
		#test_dSprite()
		test_XYS(offset=args.offset)
	else :
		queryXYS()