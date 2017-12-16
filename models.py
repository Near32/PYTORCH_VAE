import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
		outd = z_dim
		outdim = 1
		indim = dim
		pad = 0
		stride = 1
		k = int(indim +2*pad -stride*(outdim-1))
		self.fc = conv( ind, outd, k, stride=stride,pad=pad, batchNorm=False)
		self.sig = nn.Sigmoid()

	def encode(self, x) :
		out = self.cvs(x)
		out = self.sig( self.fc(out).squeeze() )
		
		return out

	def forward(self,x) :
		return self.encode(x)

class VAE(nn.Module) :
	def __init__(self, net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		#Encoder.__init__(self, img_dim=img_dim, conv_dim=conv_dim, z_dim=2*z_dim)
		#Decoder.__init__(self, img_dim=img_dim, conv_dim=conv_dim, z_dim=z_dim)
		super(VAE,self).__init__()
		self.encoder = Encoder(net_depth=net_depth,img_dim=img_dim, img_depth=img_depth,conv_dim=conv_dim, z_dim=2*z_dim)
		self.decoder = Decoder(net_depth=net_depth,img_dim=img_dim, img_depth=img_depth, conv_dim=conv_dim, z_dim=z_dim)

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
    	                                      batch_size=100, 
        	                                  shuffle=True)
	data_iter = iter(data_loader)
	iter_per_epoch = len(data_loader)

	# Model :
	z_dim = 20
	img_dim = 28
	img_depth=1
	conv_dim = 64
	use_cuda = True#False
	net_depth = 2
	vae = VAE(net_depth=net_depth,z_dim=z_dim,img_dim=img_dim,img_depth=img_depth,conv_dim=conv_dim, use_cuda=use_cuda)
	
	# Optim :
	lr = 1e-4
	optimizer = torch.optim.Adam( vae.parameters(), lr=lr)

	# Debug :
	# fixed inputs for debugging
	fixed_z = Variable(torch.randn(100, z_dim))
	if use_cuda :
		fixed_z = fixed_z.cuda()

	fixed_x, _ = next(data_iter)
	

	path = 'layers{}-z{}-conv{}-lr{}'.format(net_depth,z_dim,conv_dim,lr)
	if not os.path.exists( './data/{}/'.format(path) ) :
		os.mkdir('./data/{}/'.format(path))

	
	torchvision.utils.save_image(fixed_x.cpu(), './data/{}/real_images.png'.format(path))
	
	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim))
	if use_cuda :
		fixed_x = fixed_x.cuda()

	
	for epoch in range(50):
	    # Save the reconstructed images
	    reconst_images, _, _ = vae(fixed_x)
	    reconst_images = reconst_images.view(-1, 1, img_dim, img_dim)
	    torchvision.utils.save_image(reconst_images.data.cpu(),'./data/{}/reconst_images_{}.png'.format(path,(epoch+1)) )

	    for i, (images, _) in enumerate(data_loader):
	        
	        images = Variable( (images.view(-1,1,img_dim, img_dim) ) )
	        if use_cuda :
	        	images = images.cuda() 
	        
	        out, mu, log_var = vae(images)
	        
	        
	        # Compute reconstruction loss and kl divergence
	        # For kl_divergence, see Appendix B in the paper or http://yunjey47.tistory.com/43
	        

	        reconst_loss = F.binary_cross_entropy(out, images, size_average=False)
	        kl_divergence = torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1))
	        
	        # Backprop + Optimize
	        total_loss = reconst_loss + kl_divergence
	        optimizer.zero_grad()
	        total_loss.backward()
	        optimizer.step()
	        
	        if i % 100 == 0:
	            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
	                   "Reconst Loss: %.4f, KL Div: %.7f" 
	                   %(epoch+1, 50, i+1, iter_per_epoch, total_loss.data[0], 
	                     reconst_loss.data[0], kl_divergence.data[0]))
	    
	    

if __name__ == '__main__' :
	test_mnist()