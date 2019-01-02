"""
All the data structures used in VFPred
"""

import numpy as np

class EcgSignal(object):

	'''
		Class denoting ecg signals 		
	'''

	def __init__(self,signal,annotation,Fs,channel=None,source=None):

		self.signal = np.array(signal[:])		# signal as an array
		self.annotation = annotation			# annotation
		self.channel = channel					# channel number
		self.source = source					# name of database
		self.Fs=Fs								# sampling frequency

class Annotation(object):

	'''
		Class denoting beat anotations 		
	'''

	def __init__(self,index,label):

		self.index = index			# array of indices
		self.label = label			# array of labels

class Features(object):

	'''
		Extracted features from ecg signals 
	'''

	def __init__(self,Signal_FFT,Fs,IMF1_FFT,R_FFT,imf1_Sim,R_Sim,label,file,episode,channel,imf12,imf23):

		self.Signal_FFT = Signal_FFT	# FFT of the signal
		self.Fs = Fs					# sampling frequency
		self.IMF1_FFT=IMF1_FFT			# FFT of the IMF1
		self.R_FFT=R_FFT				# FFT of the Residue
		self.imf1_Sim=imf1_Sim			# cosine similarity of signal and IMF1
		self.R_Sim=R_Sim				# cosine similarity of signal and R
		self.label=label				# annotation label 
		self.file=file					# file name (number actually)
		self.episode=episode			# episode number
		self.channel=channel			# channel number
		self.imf12 = imf12				# cosine similarity between of IMF1 and IMF2 (unnecessary) 
		self.imf23 = imf23				# cosine similarity between of IMF2 and IMF3 (unnecessary)
