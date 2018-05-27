# Bismillahir Rahmanir Rahim
# Rabbi Zidni Ilma

import numpy as np

class EcgSignal(object):

	'''
		Class denoting ecg signals 		
	'''

	def __init__(self,signal,annotation,Fs,channel=None,source=None):

		self.signal = np.array(signal[:])
		self.annotation = annotation
		self.channel = channel
		self.source = source
		self.Fs=Fs

class Annotation(object):

	'''
		Class denoting beat anotations 		
	'''

	def __init__(self,index,label):

		self.index = index
		self.label = label

class DataV2(object):

	'''
		Extracted values from ecg signals 
	'''

	def __init__(self,Signal_FFT,Fs,IMF1_FFT,R_FFT,imf1_Sim,R_Sim,label,file,episode,channel,imf12,imf23):

		self.Signal_FFT = Signal_FFT
		self.Fs = Fs
		self.IMF1_FFT=IMF1_FFT
		self.R_FFT=R_FFT
		self.imf1_Sim=imf1_Sim
		self.R_Sim=R_Sim
		self.label=label
		self.file=file
		self.episode=episode
		self.channel=channel
		self.imf12 = imf12
		self.imf23 = imf23
