import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from math import floor

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
		dum = values.unsqueeze(0).long()
		return log_pmf.gather( 0, dum ).squeeze(0)
		
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

class STNbasedNet(nn.Module):
    def __init__(self, input_dim=256, input_depth=1, nbr_stn=2, stn_stack_input=True):
        super(STNbasedNet, self).__init__()

        self.input_dim = input_dim
        self.input_depth = input_depth
        self.nbr_stn = nbr_stn
        self.stn_stack_input=stn_stack_input

        # Spatial transformer localization-network
        stnloc = []
        dim = self.input_dim
        pad = 0
        stride = 1
        k=7
        stnloc.append( nn.Conv2d(self.input_depth, 8, kernel_size=k, padding=pad, stride=stride) )
        dim = floor( (dim-k+2*pad)/stride +1 )
        k=2
        stride = 2
        stnloc.append( nn.MaxPool2d(k, stride=stride) )
        dim = floor( (dim-k+2*pad)/stride +1 )
        stnloc.append( nn.ReLU(True) )
        k=5
        stride=1
        stnloc.append( nn.Conv2d(8, 16, kernel_size=k, padding=pad, stride=stride) )
        dim = floor( (dim-k+2*pad)/stride +1 )
        k=2
        stride = 2
        stnloc.append( nn.MaxPool2d(k, stride=stride) )
        dim = floor( (dim-k+2*pad)/stride +1 )
        stnloc.append( nn.ReLU(True) )
        self.localization = nn.Sequential( *stnloc)
        
        #print('DIM OUTPUT : {}'.format(dim) )

        # Regressor for the 3 * 2 affine matrixes :
        self.fc_loc = nn.Sequential(
            nn.Linear(16 * (dim**2), 128),
            nn.ReLU(True),
            #nn.Linear(128, self.nbr_stn * 3 * 2)
            nn.Linear(128, self.nbr_stn * 2 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        #self.fc_loc[2].weight.data += torch.rand( self.fc_loc[2].weight.size() ) * 1e-10
        #init_bias = torch.FloatTensor( [1.0, 0, 0.0, 0, 1.0, 0.0]).view((1,-1))
        init_bias = torch.FloatTensor( [1, 0, 1, 0] ).view((1,-1))
		for i in range(self.nbr_stn-1 ) :
        	#r = torch.rand( (1,6)) * 1e-10
        	r = torch.rand( (1,4)) * 1e-10
        	#ib = torch.FloatTensor( [0.5, 0, 0.0, 0, 0.5, 0.0]).view((1,-1))
        	ib = torch.FloatTensor( [0.5, 0, 0.5, 0]).view((1,-1))
        	#ib += r
        	init_bias = torch.cat( [init_bias, ib], dim=0)
        self.fc_loc[2].bias.data = init_bias.view((-1))

    # Spatial transformer network forward function
    def stn(self, x):
        batch_size = x.size()[0]
        xs = self.localization(x)
        xs = xs.view(batch_size,-1)
        theta = self.fc_loc(xs)
        #theta = theta.view(batch_size,self.nbr_stn, 2, 3)
        theta = theta.view(batch_size,self.nbr_stn, -1).contiguous()

        xd = []
        zeroft = Variable(torch.zeros((batch_size,1) ) ).cuda()
        for i in range(self.nbr_stn) :
            thetad = theta[:,i,:].contiguous()
            thetad = thetad.view((batch_size,-1,1))
            thetad = thetad.contiguous()
            thetad = torch.cat( [ thetad[:,0], zeroft, thetad[:,1], zeroft, thetad[:,2], thetad[:,3] ], dim=1)
            thetad = thetad.view((-1,2,3)).contiguous()
            grid = F.affine_grid(thetad, x.size())
            xd.append( F.grid_sample(x, grid) )

        if self.stn_stack_input :
            xd.append( x)

        xd = torch.cat( xd, dim=1)
        
        return xd

    def forward(self, x):
        batch_size = x.size()[0]
        # transform the input
        x = self.stn(x)

        return x


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


class DecoderXYS(nn.Module) :
	def __init__(self,net_depth=3, z_dim=32, img_dim=128, conv_dim=64,img_depth=3 ) :
		super(DecoderXYS,self).__init__()
		
		self.net_depth = net_depth
		self.dcs = []
		outd = conv_dim*(2**self.net_depth)
		ind= z_dim
		k = 6#4
		dim = k
		pad = 1
		stride = 2
		self.fc = deconv( ind, outd, k, stride=1, pad=0, batchNorm=False)
		
		for i in reversed(range(self.net_depth)) :
			ind = outd
			outd = 32#conv_dim*(2**i)
			self.dcs.append( deconv( ind, outd,k,stride=stride,pad=pad) )
			self.dcs.append( nn.LeakyReLU(0.05) )
			dim = k-2*pad + stride*(dim-1)
		self.dcs = nn.Sequential( *self.dcs) 
			
		ind = outd
		self.img_depth=img_depth
		outd = self.img_depth
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

class EncoderXYS(nn.Module) :
	def __init__(self,net_depth=3, img_dim=128, img_depth=3, conv_dim=64, z_dim=32 ) :
		super(EncoderXYS,self).__init__()
		
		self.net_depth = net_depth
		self.cvs = []
		outd = conv_dim
		ind= img_depth
		k = 6#4
		dim = img_dim
		pad = 1
		stride = 2
		self.cvs = []
		self.cvs.append( conv( img_depth, conv_dim, 4, batchNorm=False))
		self.cvs.append( nn.LeakyReLU(0.05) )
		dim = (dim-k+2*pad)/stride +1

		for i in range(1,self.net_depth,1) :
			ind = outd
			outd = 32#conv_dim*(2**i)
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
		# net_depth = 3 img dim 224 : net_deph = 5 img dim 128
		#self.fc = nn.Linear( 8192, 2048)
		# net_depth 5 img_dim 256 k=6 :
		self.fc = nn.Linear( 1152, 2048)
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


class betaVAEXYS(nn.Module) :
	def __init__(self, beta=1.0,net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		super(betaVAEXYS,self).__init__()
		self.encoder = EncoderXYS(z_dim=2*z_dim, img_depth=img_depth, img_dim=img_dim, conv_dim=conv_dim,net_depth=net_depth)
		self.decoder = DecoderXYS(z_dim=z_dim, img_dim=img_dim, img_depth=img_depth, net_depth=net_depth)

		self.z_dim = z_dim
		self.img_dim=img_dim
		self.img_depth=img_depth
		
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


class DecoderXYS2(nn.Module) :
	def __init__(self,net_depth=3, z_dim=32, img_dim=128, conv_dim=64,img_depth=3 ) :
		super(DecoderXYS2,self).__init__()
		
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
		self.img_depth=img_depth
		outd = self.img_depth
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

class EncoderXYS2(nn.Module) :
	def __init__(self,net_depth=3, img_dim=128, img_depth=3, conv_dim=64, z_dim=32 ) :
		super(EncoderXYS2,self).__init__()
		
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
		#self.fc = nn.Linear( 16384, 2048)
		#self.fc = nn.Linear( 32768, 2048)
		self.fc = nn.Linear( 8192, 2048)
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


class betaVAEXYS2(nn.Module) :
	def __init__(self, beta=1.0,net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		super(betaVAEXYS2,self).__init__()
		self.encoder = EncoderXYS2(z_dim=2*z_dim, img_depth=img_depth, img_dim=img_dim, conv_dim=conv_dim,net_depth=net_depth)
		self.decoder = DecoderXYS2(z_dim=z_dim, img_dim=img_dim, img_depth=img_depth, net_depth=net_depth)

		self.z_dim = z_dim
		self.img_dim=img_dim
		self.img_depth=img_depth
		
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



class DecoderXYS3(nn.Module) :
	def __init__(self,net_depth=3, z_dim=32, img_dim=128, conv_dim=64,img_depth=3 ) :
		super(DecoderXYS3,self).__init__()
		
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
		self.img_depth=img_depth
		outd = self.img_depth
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

class EncoderXYS3(nn.Module) :
	def __init__(self,net_depth=3, img_dim=128, img_depth=3, conv_dim=64, z_dim=32 ) :
		super(EncoderXYS3,self).__init__()
		
		self.net_depth = net_depth
		self.img_depth= img_depth
		self.z_dim = z_dim
		# 224
		self.cv1 = conv( self.img_depth, 96, 11, batchNorm=False)
		# 108/109 = E( (224-11+2*1)/2 ) + 1
		self.d1 = nn.Dropout2d(p=0.8)
		self.cv2 = conv( 96, 256, 5)
		# 53 / 54
		self.d2 = nn.Dropout2d(p=0.8)
		self.cv3 = conv( 256, 384, 3)
		# 27 / 27
		self.d3 = nn.Dropout2d(p=0.5)
		self.cv4 = conv( 384, 64, 1)
		# 15
		self.d4 = nn.Dropout2d(p=0.5)
		self.fc = conv( 64, 64, 4, stride=1,pad=0, batchNorm=False)
		# 12
		#self.fc1 = nn.Linear(64 * (12**2), 128)
		self.fc1 = nn.Linear(64 * (14**2), 128)
		self.bn1 = nn.BatchNorm1d(128)
		self.fc2 = nn.Linear(128, 64)
		self.bn2 = nn.BatchNorm1d(64)
		self.fc3 = nn.Linear(64, self.z_dim)

	def encode(self, x) :
		out = F.leaky_relu( self.cv1(x), 0.15)
		out = self.d1(out)
		out = F.leaky_relu( self.cv2(out), 0.15)
		out = self.d2(out)
		out = F.leaky_relu( self.cv3(out), 0.15)
		out = self.d3(out)
		out = F.leaky_relu( self.cv4(out), 0.15)
		out = self.d4(out)
		out = F.leaky_relu( self.fc(out))
		#print(out.size())
		out = out.view( -1, self.num_flat_features(out) )
		#print(out.size())
		out = F.leaky_relu( self.bn1( self.fc1( out) ), 0.15 )
		out = F.leaky_relu( self.bn2( self.fc2( out) ), 0.15)
		out = F.relu(self.fc3( out) )


		return out


	def forward(self,x) :
		return self.encode(x)

	def num_flat_features(self, x) :
		size = x.size()[1:]
		# all dim except the batch dim...
		num_features = 1
		for s in size :
			num_features *= s
		return num_features



class betaVAEXYS3(nn.Module) :
	def __init__(self, beta=1.0,net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		super(betaVAEXYS3,self).__init__()
		self.encoder = EncoderXYS3(z_dim=2*z_dim, img_depth=img_depth, img_dim=img_dim, conv_dim=conv_dim,net_depth=net_depth)
		self.decoder = DecoderXYS3(z_dim=z_dim, img_dim=img_dim, img_depth=img_depth, net_depth=net_depth)

		self.z_dim = z_dim
		self.img_dim=img_dim
		self.img_depth=img_depth
		
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

class STNbasedEncoderXYS3(STNbasedNet) :
	def __init__(self,net_depth=3, img_dim=128, img_depth=3, conv_dim=64, z_dim=32, nbr_stn=2, stn_stack_input=True ) :
		super(STNbasedEncoderXYS3,self).__init__(input_dim=img_dim, input_depth=img_depth, nbr_stn=nbr_stn, stn_stack_input=stn_stack_input)
		
		self.net_depth = net_depth
		self.img_depth= img_depth
		self.z_dim = z_dim

		self.stn_output_depth = self.input_depth*self.nbr_stn
		if self.stn_stack_input :
			self.stn_output_depth += self.input_depth

		# 224
		self.cv1 = conv( self.stn_output_depth, 96, 11, batchNorm=False)
		# 108/109 = E( (224-11+2*1)/2 ) + 1
		self.d1 = nn.Dropout2d(p=0.8)
		self.cv2 = conv( 96, 256, 5)
		# 53 / 54
		self.d2 = nn.Dropout2d(p=0.8)
		self.cv3 = conv( 256, 384, 3)
		# 27 / 27
		self.d3 = nn.Dropout2d(p=0.5)
		self.cv4 = conv( 384, 64, 1)
		# 15
		self.d4 = nn.Dropout2d(p=0.5)
		self.fc = conv( 64, 64, 4, stride=1,pad=0, batchNorm=False)
		# 12
		#self.fc1 = nn.Linear(64 * (12**2), 128)
		self.fc1 = nn.Linear(64 * (14**2), 128)
		self.bn1 = nn.BatchNorm1d(128)
		self.fc2 = nn.Linear(128, 64)
		self.bn2 = nn.BatchNorm1d(64)
		self.fc3 = nn.Linear(64, self.z_dim)

	def encode(self, x) :
		x = super(STNbasedEncoderXYS3,self).forward(x)

		out = F.leaky_relu( self.cv1(x), 0.15)
		out = self.d1(out)
		out = F.leaky_relu( self.cv2(out), 0.15)
		out = self.d2(out)
		out = F.leaky_relu( self.cv3(out), 0.15)
		out = self.d3(out)
		out = F.leaky_relu( self.cv4(out), 0.15)
		out = self.d4(out)
		out = F.leaky_relu( self.fc(out))
		#print(out.size())
		out = out.view( -1, self.num_flat_features(out) )
		#print(out.size())
		out = F.leaky_relu( self.bn1( self.fc1( out) ), 0.15 )
		out = F.leaky_relu( self.bn2( self.fc2( out) ), 0.15)
		out = F.relu(self.fc3( out) )


		return out


	def forward(self,x) :
		return self.encode(x)

	def num_flat_features(self, x) :
		size = x.size()[1:]
		# all dim except the batch dim...
		num_features = 1
		for s in size :
			num_features *= s
		return num_features


class STNbasedBetaVAEXYS3(nn.Module) :
	def __init__(self, beta=1.0,net_depth=4,img_dim=224, z_dim=32, conv_dim=64, use_cuda=True, img_depth=3) :
		super(STNbasedBetaVAEXYS3,self).__init__()
		self.encoder = STNbasedEncoderXYS3(z_dim=2*z_dim, img_depth=img_depth, img_dim=img_dim, conv_dim=conv_dim,net_depth=net_depth)
		self.decoder = DecoderXYS3(z_dim=z_dim, img_dim=img_dim, img_depth=img_depth, net_depth=net_depth)

		self.z_dim = z_dim
		self.img_dim=img_dim
		self.img_depth=img_depth
		
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



class GazeHead(nn.Module) :
	def __init__(self, outdim,nbr_latents=10, use_cuda=True):
		super(GazeHead,self).__init__()
		self.SAVE_PATH = './gazehead.weights'
		self.outdim = outdim
		self.nbr_latents = nbr_latents
		self.use_cuda = use_cuda

		self.fc1 = nn.Linear(self.nbr_latents, 256)
		self.bn1 = nn.BatchNorm1d(256)
		self.fc2 = nn.Linear(256, 128)
		self.bn2 = nn.BatchNorm1d(128)
		self.fc3 = nn.Linear(128, self.outdim)

		if self.use_cuda :
			self = self.cuda()

	def forward(self, x) :
		out = F.leaky_relu( self.bn1( self.fc1( x) ) )
		out = F.leaky_relu( self.bn2( self.fc2( out) ) )
		out = self.fc3( out)

		return out

	def setSAVE_PATH(self,path) :
		self.SAVE_PATH = path

		
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
	z_dim = 12
	img_dim = 28
	img_depth=1
	conv_dim = 32
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

	var_z0 = torch.zeros(100, z_dim)
	val = -0.5
	step = 1.0/10.0
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
	

	path = 'layers{}-z{}-conv{}-lr{}'.format(net_depth,z_dim,conv_dim,lr)
	if not os.path.exists( './data/{}/'.format(path) ) :
		os.mkdir('./data/{}/'.format(path))
	if not os.path.exists( './data/{}/gen_images/'.format(path) ) :
		os.mkdir('./data/{}/gen_images/'.format(path))

	
	torchvision.utils.save_image(fixed_x.cpu(), './data/{}/real_images.png'.format(path))
	
	fixed_x = Variable(fixed_x.view(fixed_x.size(0), img_depth, img_dim, img_dim))
	if use_cuda :
		fixed_x = fixed_x.cuda()

	
	for epoch in range(50):
	    # Save the reconstructed images
	    reconst_images, _, _ = vae(fixed_x)
	    reconst_images = reconst_images.view(-1, img_depth, img_dim, img_dim)
	    torchvision.utils.save_image(reconst_images.data.cpu(),'./data/{}/reconst_images_{}.png'.format(path,(epoch+1)) )

	    # Save generated variable images :
	    gen_images = vae.decoder(var_z0)
	    gen_images = gen_images.view(-1, img_depth, img_dim, img_dim)
	    torchvision.utils.save_image(gen_images.data.cpu(),'./data/{}/gen_images/{}.png'.format(path,(epoch+1)) )

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