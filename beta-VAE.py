import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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



class Decoder1(nn.Module) :
	def __init__(self,net_depth=3, z_dim=32, img_dim=128, conv_dim=64,img_depth=3 ) :
		super(Decoder1,self).__init__()
		
		self.net_depth = net_depth
		self.img_dim = img_dim
		self.img_depth = img_depth

		ind = z_dim
		self.fc1 = nn.Linear( ind, 1200)
		self.fc2 = nn.Linear( 1200, 1200)
		self.fc3 = nn.Linear( 1200, 1200)
		self.fc4 = nn.Linear( 1200, img_dim**2)
		

	def decode(self, z) :
		
		out = z.view( z.size(0), z.size(1))
		
		out = F.relu( self.fc1(out) )
		out = F.relu( self.fc2(out) )
		out = F.relu( self.fc3(out) )
		out = F.sigmoid( self.fc4(out))

		out = out.view( (-1, self.img_depth, self.img_dim, self.img_dim))
		
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

		'''
		ind = outd
		outd = z_dim
		outd = 64
		outdim = 1
		indim = dim
		pad = 0
		stride = 1
		k = int(indim +2*pad -stride*(outdim-1))
		self.fc = conv( ind, outd, k, stride=stride,pad=pad, batchNorm=False)
		'''
		self.fc = nn.Linear( 6272, 256)
		self.fc1 = nn.Linear( 256, z_dim)

	def encode(self, x) :
		out = self.cvs(x)
		
		out = out.view( (-1, self.num_features(out) ) )
		#print(out.size() )
		
		out = F.relu( self.fc(out) )
		out = F.relu( self.fc1(out) )
		
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



class Encoder1(nn.Module) :
	def __init__(self,net_depth=3, img_dim=128, img_depth=3, conv_dim=64, z_dim=32 ) :
		super(Encoder1,self).__init__()
		
		self.net_depth = net_depth
		
		ind= img_depth*img_dim**2
		self.fc = nn.Linear( ind, 1200)
		self.fc1 = nn.Linear( 1200, 1200)
		self.fc2 = nn.Linear( 1200, 1200)
		self.fc3 = nn.Linear( 1200, z_dim)

	def encode(self, x) :
		
		out = x.view( (-1, self.num_features(x) ))

		out = F.relu( self.fc(out) )
		out = F.relu( self.fc1(out) )
		out = F.relu( self.fc2(out) )
		out = F.relu( self.fc3(out) )
		
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


class Encoder2(nn.Module) :
	def __init__(self,net_depth=3, img_dim=128, img_depth=3, conv_dim=64, z_dim=32 ) :
		super(Encoder2,self).__init__()
		
		self.net_depth = net_depth
		
		outd = conv_dim
		ind= img_depth
		k = 4
		dim = img_dim
		
		pad = 0
		stride = 2
		
		self.cv1 = conv( img_depth, conv_dim, k, batchNorm=False)
		self.cv2 = conv( conv_dim, conv_dim, k,stride=stride,pad=pad)
		self.cv3 = conv( conv_dim, conv_dim*2, k,stride=stride,pad=pad)
		
		self.fc = nn.Linear( 512, 256)
		self.fc1 = nn.Linear( 256, z_dim)

	def encode(self, x) :
		
		out = F.relu( self.cv1(x) )
		out = F.relu( self.cv2(out) )
		out = F.relu( self.cv3(out) )
		
		out = out.view( (-1, self.num_features(out) ) )
		#print(out.size() )
		
		out = F.relu( self.fc(out) )
		out = F.relu( self.fc1(out) )
		
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
		self.encoder = Encoder1(net_depth=net_depth,img_dim=img_dim, img_depth=img_depth,conv_dim=conv_dim, z_dim=2*z_dim)
		self.decoder = Decoder1(net_depth=net_depth,img_dim=img_dim, img_depth=img_depth, conv_dim=conv_dim, z_dim=z_dim)

		self.beta = beta
		self.use_cuda = use_cuda

		if self.use_cuda :
			self = self.cuda()

	def reparameterize(self, mu,log_var) :
		eps = torch.randn( (mu.size()[0], mu.size()[1]) )
		veps = Variable( eps)
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

def test_mnist():
	import os
	import torchvision
	from torchvision import datasets, transforms
	dataset = datasets.MNIST(root='./data',
		                     train=True,
		                     transform=transforms.ToTensor(),
							 download=True) 

	# Data loader
	data_loader = torch.utils.data.DataLoader(dataset=dataset,
    	                                      batch_size=32, 
        	                                  shuffle=True)
	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Model :
	z_dim = 100
	img_dim = 28
	img_depth=1
	conv_dim = 64
	use_cuda = True#False
	net_depth = 2
	beta = 1.0
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

	var_z0 = torch.zeros(100, z_dim)
	val = -1.0
	step = 2.0/10.0
	for i in range(10) :
		var_z0[i,0] = val
		var_z0[i+10,1] = val
		var_z0[i+20,2] = val
		var_z0[i+30,3] = val
		var_z0[i+40,4] = val
		var_z0[i+50,5] = val
		var_z0[i+60,6] = val
		var_z0[i+70,7] = val
		var_z0[i+80,8] = val
		var_z0[i+90,9] = val
		val += step

	var_z0 = Variable(var_z0)
	if use_cuda :
		var_z0 = var_z0.cuda()

	fixed_x, _ = next(data_iter)
	

	path = 'beta{}-layers{}-z{}-conv{}-lr{}'.format(beta,net_depth,z_dim,conv_dim,lr)
	if not os.path.exists( './beta-data/{}/'.format(path) ) :
		os.mkdir('./beta-data/{}/'.format(path))
	if not os.path.exists( './beta-data/{}/gen_images/'.format(path) ) :
		os.mkdir('./beta-data/{}/gen_images/'.format(path))

	
	torchvision.utils.save_image(fixed_x.cpu(), './beta-data/{}/real_images.png'.format(path))
	
	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim))
	if use_cuda :
		fixed_x = fixed_x.cuda()

	out = torch.zeros((1,1))

	for epoch in range(50):
	    # Save the reconstructed images
	    reconst_images, _, _ = betavae(fixed_x)
	    reconst_images = reconst_images.view(-1, img_depth, img_dim, img_dim)
	    torchvision.utils.save_image(reconst_images.data.cpu(),'./beta-data/{}/reconst_images_{}.png'.format(path,(epoch+1)) )

	    # Save generated variable images :
	    gen_images = betavae.decoder(var_z0)
	    gen_images = gen_images.view(-1, img_depth, img_dim, img_dim)
	    torchvision.utils.save_image(gen_images.data.cpu(),'./beta-data/{}/gen_images/{}.png'.format(path,(epoch+1)) )

	    #print(out)

	    for i, (images, _) in enumerate(data_loader):
	        
	        images = Variable( (images.view(-1,1,img_dim, img_dim) ) )
	        if use_cuda :
	        	images = images.cuda() 
	        
	        out, mu, log_var = betavae(images)
	        
	        # Compute :
	        #reconstruction loss :
	        reconst_loss = F.binary_cross_entropy(out, images, size_average=False)
	        #reconst_loss = torch.mean( (out.view(-1) - images.view(-1))**2 )
	        
	        # expected log likelyhood :
	        expected_log_lik = torch.mean( Bernoulli( out.view((-1)) ).log_prob( images.view((-1)) ) )
	        # kl divergence :
	        kl_divergence = 0.5 * torch.mean( torch.sum( (mu**2 + torch.exp(log_var) - log_var -1), dim=1) )
	        
	        # Backprop + Optimize
	        #total_loss = reconst_loss + betavae.beta*kl_divergence
	        #total_loss = reconst_loss
	        elbo = expected_log_lik - betavae.beta * kl_divergence
	        total_loss = -elbo

	        optimizer.zero_grad()
	        total_loss.backward()
	        optimizer.step()
	        
	        if i % 100 == 0:
	            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
	                   "Reconst Loss: %.4f, KL Div: %.7f, E[logp]: %.7f" 
	                   %(epoch+1, 50, i+1, iter_per_epoch, total_loss.data[0], 
	                     reconst_loss.data[0], kl_divergence.data[0],expected_log_lik.data[0]))
	    
	    

if __name__ == '__main__' :
	test_mnist()