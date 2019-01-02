'''
Codes for signal processing
'''

from PyEMD import EMD
import pickle
import matplotlib.pyplot as plt
from data_structures import EcgSignal, Features
import numpy as np
import os.path
from scipy.fftpack import fft , fftshift
import seaborn
from helper_function import cosineSimilarity
from tqdm import tqdm

def filtering(signal,Fs,step2='movingAverage',step2Param=5,highPassFc=1,lowPassFc=30,butterworthOrder=12):
	"""
	Filters and preprocesses the signal.
		
		1 ) Substract the mean
		2 ) Moving average for high order noises
		3 ) High pass filter for drift suppression
		4 ) Low pass filter for high frequency suppression

	As presented in 
						E.M. Abu Anas, S.Y. Lee, M.K. Hasan
						Exploiting correlation of ECG with certain emd functions for discrimination of ventricular fibrillation
						Comput. Biol. Med., 41 (2) (2011), pp. 110-114
	
	Arguments:
		signal {numpy array} -- signal
		Fs {int} -- sampling frequency
	
	Keyword Arguments:
		step2 {str} -- which mode of filtering is used ? 'movingAverage' or 'gaussianFilter' (default: {'movingAverage'})
		step2Param {int} -- window length (default: {5})
		highPassFc {int} -- cut-off freq for high pass filter (default: {1})
		lowPassFc {int} -- cut-off freq for low pass filter (default: {30})
		butterworthOrder {int} -- order of butter worth filter (default: {12})
	
	Returns:
		[{numpy array}] -- the filtered and processed signal
	"""

	from scipy.signal import butter, filtfilt, lfilter

	# 1 ) Substract the mean

	miu = np.mean(signal)		# mean of the signal

	signal -= miu 


	# 2 ) Moving average for high order noises

	if(step2=='movingAverage'):

		signal2 = np.convolve(signal,np.ones(step2Param)/(step2Param*1.0))		# equivalent to moving average


	elif(step2=='gaussianFilter'):

		# to be implemented later 
		pass 

	# 3 ) High pass filter for drift suppression

	Fnyq = 0.5*Fs			# nyquist frequency 

	b, a = butter(1, highPassFc/ Fnyq , btype='highpass')		# definition of the high pass filter
	signal3 = lfilter(b, a, signal2)

	# 4 ) Low pass filter for high frequency suppression

	b,a = butter(12,lowPassFc/Fnyq)								# definition of the high pass filter
	signal4 = lfilter(b, a, signal3)
	
	return signal4


def processSignal(db,signal,Fs,plot,label,file,episode,channel=1,Te=5):
	"""
	Extracts features from the signal
	
	Arguments:
		db {str} -- name of the database
		signal {array} -- signal
		Fs {int} -- sampling frequency
		plot {bool} -- plot or not ?
		label {[type]} -- annotation
		file {int} -- file number
		episode {int} -- episode number
	
	Keyword Arguments:
		channel {int} -- channel number (default: {1})
		Te {int} -- episode length (default: {5})
	"""



	Le = 5			# need to ignore this for now
	La = Te			# episode length
	
	alpha = 0.05	# parameters used in paper
	beta = 0.02	

	delL = Le - La + 1		# ignore this for now, should be 1

	nSamplesInWindow = Te * Fs		# samples in a window

	thetaIMF = 0.0		# 
	thetaR = 0.0
	theta12 = 0.0
	theta23 = 0.0

	# (1) Choose a segment xi(n) of ECG signal of duration Te-sec containing N samples.

	window = signal[:]	

	window2 = filtering(signal=window,Fs=Fs,lowPassFc=20)		# filtering and preprocessing

	# computing Empirical Mode Decomposition	

	IMF = EMD().emd(window2)
	IMF1 = IMF[0]		# first IMF 
			
	Vn = alpha * np.max(window2)		# please refer to paper for this section

	sumIMF1sq = 0			
	sumXsq = 0

	for j in range(len(window2)):

		if(abs(IMF1[j])<=Vn):

			sumIMF1sq += IMF1[j]**2
			sumXsq += window2[j] ** 2

	if(abs(sumXsq-0.0)<1e-7):
		return						# there is no valid ecg signal, just flat line signal

	NLCR = sumIMF1sq / sumXsq		

	if(NLCR<=beta):		# still noisy

		IMF1 = IMF[0] + IMF[1]

		try:						# (this is unncessary now-)
			IMF2 = IMF[1]
		except:
			IMF2 = np.zeros(len(IMF1))

		try:						# (this is unncessary now-)
			IMF3 = IMF[2]
		except:
			IMF3 = np.zeros(len(IMF1))

		R = window2 - IMF1			# computing residue

	else:				# less noisy

		IMF1 = IMF[0]

		try:						# (this is unncessary now-)
			IMF2 = IMF[1]
		except:
			IMF2 = np.zeros(len(IMF1))

		try:						# (this is unncessary now-)
			IMF3 = IMF[2]
		except:
			IMF3 = np.zeros(len(IMF1))

		R = window2 - IMF1 - IMF2	# computing residue

		
	Signal_FFT = fftshift(fft(window2))			# computing FFT of signal
	IMF1_FFT_CMPLX = fftshift(fft(IMF1))		# computing FFT of IMF1
	R_FFT_CMPLX = fftshift(fft(R))				# computing FFT of Residue
	IMF1_Sim = cosineSimilarity(window2,IMF1)	# cosine similarity of signal and IMF1 (unnecessary) 
	R_Sim = cosineSimilarity(window2, R)		# cosine similarity of signal and R (unnecessary) 
	IMF12 = cosineSimilarity(IMF1,IMF2)			# cosine similarity between of IMF1 and IMF2 (unnecessary) 
	IMF23 = cosineSimilarity(IMF2, IMF3)		# cosine similarity between of IMF2 and IMF3 (unnecessary) 

					# extracted featrues
	newData = Features(Signal_FFT=Signal_FFT,Fs=Fs,IMF1_FFT=IMF1_FFT_CMPLX, R_FFT=R_FFT_CMPLX, imf1_Sim=IMF1_Sim, R_Sim=R_Sim, label=label, file=file, episode=episode, channel=channel,imf12=IMF12,imf23=IMF23)
					
					# saving the features
	saveToPickle(db,data=newData,file=file,episode=episode,channel=channel)
	

def batchSignalProcessing(Te=5):
	"""
	Process all the ecg signals
	
	Keyword Arguments:
		Te {int} -- episode length (default: {5})
	"""


	try:					# creating necessary directories

		os.mkdir('Pickles/MITMVAdbFFT')
		os.mkdir('Pickles/cudbFFT')

	except:

		pass

	for fileName in tqdm(range(418, 700)):		# all the mitMVA db files

		for j in (range(2300)):			# all the episodes

			if (os.path.isfile("Pickles/MITMVAdb/" + str(fileName) + "E" + str(j) + "C" + str(1) + ".p")):

				# load the ecg episode
				ecgEpisode = pickle.load(open("Pickles/MITMVAdb/" + str(fileName) + "E" + str(j) + "C" + str(1) + ".p", "rb"))
							
				# process the ecg episode
				processSignal('MITMVA',ecgEpisode.signal, ecgEpisode.Fs, False, label=ecgEpisode.annotation, file=fileName, episode=j, channel=1,Te=Te)


	for fileName in tqdm(range(40)):		# all the cudb files

		for j in (range(2200)):			# all the episodes

			if (os.path.isfile("Pickles/cudb/" + str(fileName) + "E" + str(j) + "C" + str(1) + ".p")):

				# load the ecg episode
				ecgEpisode = pickle.load(open("Pickles/cudb/" + str(fileName) + "E" + str(j) + "C" + str(1) + ".p", "rb"))

				# process the ecg episode
				processSignal('cudb',ecgEpisode.signal, ecgEpisode.Fs, False, label=ecgEpisode.annotation, file=fileName, episode=j, channel=1,Te=Te)


def saveToPickle(db,data,file,episode,channel=1):
	"""
	Save the features as a pickle file
	
	Arguments:
		db {str} -- name of the db
		data {Feature} -- extracted Features
		file {int} -- file number
		episode {int} -- episode number
	
	Keyword Arguments:
		channel {int} -- channel number (default: {1})
	"""

	if(db=='MITMVA'):			# save in mitMVA db directory
		pickle.dump(data, open("Pickles/MITMVAdbFFT/F" + str(file) + "E" + str(episode) + "C"+str(channel)+".p", "wb"))
	
	elif(db=='cudb'):			# save in cudb directory
		pickle.dump(data, open("Pickles/cudbFFT/F" + str(file) + "E" + str(episode) + "C" + str(channel) + ".p", "wb"))
	