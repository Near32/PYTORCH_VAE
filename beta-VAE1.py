import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from skimage import io, transform
import numpy as np

from PIL import Image
from visualize import save_dot


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
		out = self.fc(z)
		out = self.dcs(out)
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
		self.fc = nn.Linear( 8192, 2048)
		self.fc1 = nn.Linear( 2048, 1024)
		self.fc2 = nn.Linear( 1024, z_dim)
		
	def encode(self, x) :
		out = self.cvs(x)

		out = out.view( (-1, self.num_features(out) ) )
		#print(out.size() )

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
	

	path = 'beta{}-layers{}-z{}-conv{}-lr{}'.format(beta,net_depth,z_dim,conv_dim,lr)
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
			#expected_log_lik = torch.mean( Bernoulli( out.view((-1)) ).log_prob( images.view((-1)) ) )
			expected_log_lik = torch.mean( Bernoulli( out ).log_prob( images ) )

			# kl divergence :
			#kl_divergence = 0.5 * torch.mean( torch.sum( (mu**2 + torch.exp(log_var) - log_var -1), dim=1) )
			kl_divergence = 0.5 * torch.sum( (mu**2 + torch.exp(log_var) - log_var -1) )

			#save_dot(kl_divergence, './kl.pdf')
			#save_dot(mu, './mu.pdf')
			#save_dot(expected_log_lik, './exp_log_likelyhood.pdf')
			#save_dot(reconst_loss, './reconst_loss.pdf')

			#print(reconst_loss,expected_log_lik.exp(),expected_log_lik_out.exp(),expected_log_lik_images.exp())

			# Backprop + Optimize
			total_loss = reconst_loss + betavae.beta*kl_divergence
			#total_loss = reconst_loss
			elbo = expected_log_lik - betavae.beta * kl_divergence
			#total_loss = -elbo

			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()

			if i % 100 == 0:
			    print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
			           "Reconst Loss: %.4f, KL Div: %.7f, E[ |~| p(x|theta)]: %.7f " 
			           %(epoch+1, 50, i+1, iter_per_epoch, total_loss.data[0], 
			             reconst_loss.data[0], kl_divergence.data[0],expected_log_lik.exp().data[0]) )



if __name__ == '__main__' :
	test_mnist()