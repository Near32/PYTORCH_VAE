import numpy as np 
import pandas as pd
import os

def csv2dict(csv) :
	ret = dict()
	for key in csv.columns :
		if not( 'Unnamed' in key) :
			column = csv.ix[:,key]
			shape = column.shape
			ret[key] = [ csv.ix[i,key] for i in range(shape[0]) ]
	return ret


class statsLogger :
	def __init__(self,path='./',filename='logs.csv',saveT=1) :
		self.path = path
		self.filename = filename
		self.saveT = saveT
		self.counterSaveT = 0

		if not os.path.exists(self.path) :
			os.mkdir(self.path)

		self.data = dict()
		self.csv = None 
		if self.filename in os.listdir(self.path) :
			self.csv = pd.read_csv(os.path.join(self.path,self.filename) )
			self.data =  csv2dict(self.csv)
			self.csv = pd.DataFrame(self.data, columns = self.data.keys()) 
		else :
			self.csv = pd.DataFrame(self.data, columns = self.data.keys())
			self.csv.to_csv( os.path.join(self.path,self.filename) )

	def save(self) :
		self.regularizeData()
		self.csv = pd.DataFrame(self.data, columns = self.data.keys())
		self.csv.to_csv(os.path.join(self.path,self.filename) )

	def show(self) :
		self.regularizeData()
		self.csv = pd.DataFrame(self.data, columns = self.data.keys())
		print(self.csv)

	def append(self,x) :
		for key in x.keys() :
			if not(key in self.data.keys()) :
				self.data[key] = []
			for el in x[key] :
				self.data[key].append(el)

		self.counterSaveT = (self.counterSaveT+1)%self.saveT
		if self.counterSaveT == 0 :
			self.save()

	def regularizeData(self) :
		#regularize dimensions :
		# 
		maxdim = 0
		for key in self.data.keys() :
			dim = len(self.data[key] )
			if dim > maxdim :
				maxdim = dim
		#
		for key in self.data.keys() :
			dim = len(self.data[key] )
			for i in range(maxdim-dim) : self.data[key].append('NaN')

def test_csv() :
	sl = statsLogger()
	sl.show()
	new = {'episodes':[0,1,2,3,4,5],'reward':[0,0,1,3,5]}
	sl.append(new)
	sl.show()
	sl.save()

if __name__ == '__main__' :
	test_csv()