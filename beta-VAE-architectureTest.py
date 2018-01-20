import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from skimage import io, transform
import numpy as np

from PIL import Image


class Distribution(object) :
	def sample(self) :
		raise NotImplementedError

	def log_prob(self,values) :
		raise NotImplementedError

class Bernoulli(Distribution) :
	def __init__(self, probs) :
		self.probs = probs

	def sample(self) :
		return torch.bernoulli(self.probs)

	def log_prob(self,values) :
		log_pmf = ( torch.stack( [1-self.probs, self.probs] ) ).log()
		return log_pmf.gather( 0, values.unsqueeze(0).long() ).squeeze(0)
		
		#logits, value = broadcast_all(self.probs, values)
		#return -F.binary_cross_entropy_with_logits(logits, value, reduce=False)

		#return -F.binary_cross_entropy_with_logits(self.probs, values)

def conv( sin, sout,k,stride=2,pad=1,batchNorm=True) :
	layers = []
	layers.append( nn.Conv2d( sin,sout, k, stride,pad) )
	if batchNorm :
		layers.append( nn.BatchNorm2d( sout) )
	return nn.Sequential( *layers )

def deconv( sin, sout,k,stride=2,pad=1,batchNorm=True) :
	layers = []
	layers.append( nn.ConvTranspose2d( sin,sout, k, stride,pad) )
	if batchNorm :
		layers.append( nn.BatchNorm2d( sout) )
	return nn.Sequential( *layers )

class Decoder(nn.Module) :
	def __init__(self,net_depth=3, z_dim=32, img_dim=128, conv_dim=64,img_depth=3 ) :
		super(Decoder,self).__init__()
		
		self.net_depth = net_depth
		self.dcs = []
		outd = conv_dim*(2**self.net_depth)
		ind= z_dim
		k = 4
		dim = k
		pad = 1
		stride = 2
		self.fc = deconv( ind, outd, k, stride=1, pad=0, batchNorm=False)
		
		for i in reversed(range(self.net_depth)) :
			ind = outd
			outd = conv_dim*(2**i)
			self.dcs.append( deconv( ind, outd,k,stride=stride,pad=pad) )
			self.dcs.append( nn.LeakyReLU(0.05) )
			dim = k-2*pad + stride*(dim-1)
		self.dcs = nn.Sequential( *self.dcs) 
			
		ind = outd
		outd = 1
		outdim = img_dim
		indim = dim
		pad = 0
		stride = 1
		k = outdim +2*pad -stride*(indim-1)
		self.dcout = deconv( ind, outd, k, stride=stride, pad=pad, batchNorm=False)
		
	def decode(self, z) :
		z = z.view( z.size(0), z.size(1), 1, 1)
		out = F.leaky_relu( self.fc(z), 0.05)
		out = F.leaky_relu( self.dcs(out), 0.05)
		out = F.sigmoid( self.dcout(out))
		return out

	def forward(self,z) :
		return self.decode(z)

class Encoder(nn.Module) :
	def __init__(self,net_depth=3, img_dim=128, img_depth=3, conv_dim=64, z_dim=32 ) :
		super(Encoder,self).__init__()
		
		self.net_depth = net_depth
		self.cvs = []
		outd = conv_dim
		ind= img_depth
		k = 4
		dim = img_dim
		pad = 1
		stride = 2
		self.cvs = []
		self.cvs.append( conv( img_depth, conv_dim, 4, batchNorm=False))
		self.cvs.append( nn.LeakyReLU(0.05) )
		dim = (dim-k+2*pad)/stride +1

		for i in range(1,self.net_depth,1) :
			ind = outd
			outd = conv_dim*(2**i)
			self.cvs.append( conv( ind, outd,k,stride=stride,pad=pad) )
			self.cvs.append( nn.LeakyReLU(0.05) )
			dim = (dim-k+2*pad)/stride +1
		self.cvs = nn.Sequential( *self.cvs)

		ind = outd
		outd = 64
		outdim = 1
		indim = dim
		pad = 0
		stride = 1
		#k = int(indim +2*pad -stride*(outdim-1))
		k=4
		
		#self.fc = conv( ind, outd, k, stride=stride,pad=pad, batchNorm=False)
		# net_depth = 5 :
		#self.fc = nn.Linear( 25088, 2048)
		# net_depth = 3 :
		self.fc = nn.Linear( 8192, 2048)
		self.fc1 = nn.Linear( 2048, 1024)
		self.fc1 = nn.Linear( 2048, 1024)
		self.fc2 = nn.Linear( 1024, z_dim)
		
	def encode(self, x) :
		out = self.cvs(x)

		out = out.view( (-1, self.num_features(out) ) )
		#print(out.size() )

		out = F.leaky_relu( self.fc(out), 0.05 )
		out = F.leaky_relu( self.fc1(out), 0.05 )
		out = self.fc2(out)
		
		return out

	def forward(self,x) :
		return self.encode(x)

	def num_features(self, x) :
		size = x.size()[1:]
		# all dim except the batch dim...
		num_features = 1
		for s in size :
			num_features *= s
		return num_features

class betaVAE(nn.Module) :
	def __init__(self, beta=1.0,net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		super(betaVAE,self).__init__()
		self.encoder = Encoder(net_depth=net_depth,img_dim=img_dim, img_depth=img_depth,conv_dim=conv_dim, z_dim=2*z_dim)
		self.decoder = Decoder(net_depth=net_depth,img_dim=img_dim, img_depth=img_depth, conv_dim=conv_dim, z_dim=z_dim)

		self.beta = beta
		self.use_cuda = use_cuda

		if self.use_cuda :
			self = self.cuda()

	def reparameterize(self, mu,log_var) :
		eps = torch.randn( (mu.size()[0], mu.size()[1]) )
		veps = Variable( eps)
		#veps = Variable( eps, requires_grad=False)
		if self.use_cuda :
			veps = veps.cuda()
		z = mu + veps * torch.exp( log_var/2 )
		return z

	def forward(self,x) :
		h = self.encoder( x)
		mu, log_var = torch.chunk(h, 2, dim=1 )
		z = self.reparameterize( mu,log_var)
		out = self.decoder(z)

		return out, mu, log_var

class DecoderdSprite(nn.Module) :
	def __init__(self,z_dim=32, img_dim=128,img_depth=3 ) :
		super(DecoderdSprite,self).__init__()
		
		self.img_dim = img_dim
		self.img_depth = img_depth

		self.fc = nn.Linear( z_dim, 1200)
		self.fc1 = nn.Linear( 1200, 1200)
		self.fc2 = nn.Linear( 1200, 1200)
		self.fc3 = nn.Linear( 1200, 4096)
		
	def decode(self, x) :
		
		out = F.tanh( self.fc(x) )
		out = F.tanh( self.fc1(out) )
		out = F.tanh( self.fc2(out) )
		self.out_logits = self.fc3(out).view( (-1,self.img_depth,self.img_dim, self.img_dim) )
		out = F.sigmoid( self.out_logits )
		

		return out

	def forward(self,z) :
		return self.decode(z)

class EncoderdSprite(nn.Module) :
	def __init__(self, z_dim=310 ) :
		super(EncoderdSprite,self).__init__()
		
		self.fc = nn.Linear( 4096, 1200)
		self.fc1 = nn.Linear( 1200, 1200)
		self.fc2 = nn.Linear( 1200, z_dim)
		
	def encode(self, x) :
		out = x.view( (-1, self.num_features(x) ) )
		
		out = F.relu( self.fc(out) )
		out = F.relu( self.fc1(out) )
		out = F.relu( self.fc2(out) )
		
		return out

	def forward(self,x) :
		return self.encode(x)

	def num_features(self, x) :
		size = x.size()[1:]
		# all dim except the batch dim...
		num_features = 1
		for s in size :
			num_features *= s
		return num_features

class betaVAEdSprite(nn.Module) :
	def __init__(self, beta=1.0,net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		super(betaVAEdSprite,self).__init__()
		self.encoder = EncoderdSprite(z_dim=2*z_dim)
		self.decoder = DecoderdSprite(z_dim=z_dim, img_dim=img_dim, img_depth=img_depth)

		self.beta = beta
		self.use_cuda = use_cuda

		if self.use_cuda :
			self = self.cuda()

	def reparameterize(self, mu,log_var) :
		eps = torch.randn( (mu.size()[0], mu.size()[1]) )
		veps = Variable( eps)
		#veps = Variable( eps, requires_grad=False)
		if self.use_cuda :
			veps = veps.cuda()
		z = mu + veps * torch.exp( log_var/2 )
		return z

	def forward(self,x) :
		h = self.encoder( x)
		mu, log_var = torch.chunk(h, 2, dim=1 )
		z = self.reparameterize( mu,log_var)
		out = self.decoder(z)

		return out, mu, log_var


class Rescale(object) :
	def __init__(self, output_size) :
		assert( isinstance(output_size, (int, tuple) ) )
		self.output_size = output_size

	def __call__(self, sample) :
		image = sample
		#image = np.array( sample )
		#h,w = image.shape[:2]

		new_h, new_w = self.output_size

		#img = transform.resize(image, (new_h, new_w) )
		img = image.resize( (new_h,new_w) ) 

		sample = np.reshape( img, (1, new_h, new_w) )

		return sample 

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
	batch_size = 128
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
	beta = 4e0
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
		torchvision.utils.save_image(255.0*gen_images,'./beta-data/{}/gen_images/{}.png'.format(path,(epoch+32)) )

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
				torchvision.utils.save_image(255*ri,'./beta-data/{}/reconst_images/{}.png'.format(path,(epoch+32)) )
			
			
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

if __name__ == '__main__' :
	#test_mnist()
	test_dSprite()