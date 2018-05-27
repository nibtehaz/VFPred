# Bismillahir Rahmanir Rahim
# Rabbi Zidni Ilma


from PyEMD import EMD
import pickle
import matplotlib.pyplot as plt
from dataStructures import EcgSignal, DataV2
import numpy as np
import os.path
from scipy.fftpack import fft , fftshift
import seaborn
from helperFunction import cosineSimilarity
from tqdm import tqdm

def filtering(signal,Fs,step2='movingAverage',step2Param=5,highPassFc=1,lowPassFc=30,butterworthOrder=12):

	'''
		Signal is NP array

		1 ) Substract the mean
		2 ) Moving average for high order noises
		3 ) High pass filter for drift suppression
		4 ) Low pass filter for high frequency suppression
	'''

	from scipy.signal import butter, filtfilt, lfilter

	# 1 ) Substract the mean

	miu = np.mean(signal)

	signal -= miu 


	# 2 ) Moving average for high order noises

	if(step2=='movingAverage'):

		signal2 = np.convolve(signal,np.ones(step2Param)/(step2Param*1.0))


	elif(step2=='gaussianFilter'):

		# to be implemented later 
		pass 

	# 3 ) High pass filter for drift suppression

	Fnyq = 0.5*Fs

	b, a = butter(1, highPassFc/ Fnyq , btype='highpass')
	signal3 = lfilter(b, a, signal2)

	# 4 ) Low pass filter for high frequency suppression


	b,a = butter(12,lowPassFc/Fnyq)
	signal4 = lfilter(b, a, signal3)
	
	return signal4


def processSignal(db,signal,Fs,plot,label,file,episode,channel=1,Te=5):


	Le = 5
	La = Te
	alpha = 0.05
	beta = 0.02

	delL = Le - La + 1

	nSamplesInWindow = Te * Fs

	thetaIMF = 0.0
	thetaR = 0.0
	theta12 = 0.0
	theta23 = 0.0

	# (1) Choose a segment xi(n) of ECG signal of duration Te-sec containing N samples.

	window = signal[:]

	window2 = filtering(signal=window,Fs=Fs,lowPassFc=20)

	

	IMF = EMD().emd(window2)
	IMF1 = IMF[0]
			
	Vn = alpha * np.max(window2)

	sumIMF1sq = 0
	sumXsq = 0

	for j in range(len(window2)):

		if(abs(IMF1[j])<=Vn):

			sumIMF1sq += IMF1[j]**2
			sumXsq += window2[j] ** 2

	if(abs(sumXsq-0.0)<1e-7):
		return

	NLCR = sumIMF1sq / sumXsq

	if(NLCR<=beta):

		IMF1 = IMF[0] + IMF[1]

		try:
			IMF2 = IMF[1]
		except:
			IMF2 = np.zeros(len(IMF1))

		try:
			IMF3 = IMF[2]
		except:
			IMF3 = np.zeros(len(IMF1))

		R = window2 - IMF1

	else:

		IMF1 = IMF[0]

		try:
			IMF2 = IMF[1]
		except:
			IMF2 = np.zeros(len(IMF1))

		try:
			IMF3 = IMF[2]
		except:
			IMF3 = np.zeros(len(IMF1))

		R = window2 - IMF1 - IMF2

		

	# processed signal fft
	Signal_FFT = fftshift(fft(window2))
	Fs = Fs
	IMF1_FFT_CMPLX = fftshift(fft(IMF1))
	R_FFT_CMPLX = fftshift(fft(R))
	IMF1_Sim = cosineSimilarity(window2,IMF1)
	R_Sim = cosineSimilarity(window2, R)
	IMF12 = cosineSimilarity(IMF1,IMF2)
	IMF23 = cosineSimilarity(IMF2, IMF3)
	newData = DataV2(Signal_FFT=Signal_FFT,Fs=Fs,IMF1_FFT=IMF1_FFT_CMPLX, R_FFT=R_FFT_CMPLX, imf1_Sim=IMF1_Sim, R_Sim=R_Sim, label=label, file=file, episode=episode, channel=channel,imf12=IMF12,imf23=IMF23)

	saveToPickle(db,data=newData,file=file,episode=episode,channel=channel)

def batchSignalProcessing(Te=5):

	try:

		os.mkdir('Pickles/MITMVAdbFFT')
		os.mkdir('Pickles/cudbFFT')

	except:

		pass

	for fileName in tqdm(range(418, 700)):

		break

		for j in (range(2300)):

			if (os.path.isfile("Pickles/MITMVAdb/" + str(fileName) + "E" + str(j) + "C" + str(1) + ".p")):
				ecgEpisode = pickle.load(open("Pickles/MITMVAdb/" + str(fileName) + "E" + str(j) + "C" + str(1) + ".p", "rb"))


				processSignal('MITMVA',ecgEpisode.signal, ecgEpisode.Fs, False, label=ecgEpisode.annotation, file=fileName, episode=j, channel=1,Te=Te)


	for fileName in tqdm(range(40)):

		for j in (range(2200)):

			if (os.path.isfile("Pickles/cudb/" + str(fileName) + "E" + str(j) + "C" + str(1) + ".p")):
				ecgEpisode = pickle.load(open("Pickles/cudb/" + str(fileName) + "E" + str(j) + "C" + str(1) + ".p", "rb"))

				
				processSignal('cudb',ecgEpisode.signal, ecgEpisode.Fs, False, label=ecgEpisode.annotation, file=fileName, episode=j, channel=1,Te=Te)

			j += 1

def saveToPickle(db,data,file,episode,channel=1):


	if(db=='MITMVA'):
		pickle.dump(data, open("Pickles/MITMVAdbFFT/F" + str(file) + "E" + str(episode) + "C"+str(channel)+".p", "wb"))
	elif(db=='cudb'):
		pickle.dump(data, open("Pickles/cudbFFT/F" + str(file) + "E" + str(episode) + "C" + str(channel) + ".p", "wb"))
	