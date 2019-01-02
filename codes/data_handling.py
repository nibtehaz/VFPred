"""
Downloads data from Physionet
"""

import wfdb
import os 
import pickle
import numpy as np
from tqdm import tqdm
from data_structures import EcgSignal, Annotation

def createLabelsDict(VF,notVF):
	"""
	Return a numpy array of lenth 10
	
	Arguments:
		VF {int} -- number of VF samples
		notVF {int} -- number of not VF samples
	
	Returns:
		[numpy array] -- 
						1st element is 1 if atleast 10% of the samples are VF
						2nd element is 1 if atleast 20% of the samples are VF
						...
						...
						...
						10th element is 1 if atleast 100% of the samples are VF
	"""

	li = []			# output array

	tot = VF + notVF		# total samples

	for i in range(1,11):

		if(VF>=((tot*i)//10)):		# compute ratio

			li.append(1)			# VF labeled

		else:
			li.append(0)			# not VF labeled

	return np.array(li)


def createAnnotationArray(indexArray,labelArray,hi,NSRsymbol):
	'''
	Create the annotation array
	
	Arguments:
		indexArray {list} -- list of indices
		labelArray {list} -- list of labels
		hi {int} -- length
		NSRsymbol {str} -- string representation of NSR
	
	Returns:
		[list] -- point wise annotation array
	'''


	annotations = []

	for i in range(len(indexArray)):

		annotations.append(Annotation(index=indexArray[i],label=labelArray[i]))

	distributedAnnotations = createDistributedAnnotations(annotationArray=annotations,hi=hi,NSRsymbol=NSRsymbol)

	return distributedAnnotations


def createDistributedAnnotations(annotationArray,hi,NSRsymbol):
	"""
	Generate pointwise annotation
	
	Arguments:
		annotationArray {array} -- array of annotations
		hi {int} -- length
		NSRsymbol {str} -- string representation of NSR
	
	Returns:
		[list] -- point wise annotation array
	"""


	labelArray=[]

	localLo = 0
	localHi = annotationArray[0].index
	currLabel = NSRsymbol

	## The following is similar to interval covering algorithms

	## We are assuming the first unannotated part to be NSR

	for i in range(localLo,localHi):

		labelArray.append(currLabel)


	## now for the other actual annotated segments

	for i in range(1,len(annotationArray)):			
													# interval
		localLo = annotationArray[i-1].index
		localHi = annotationArray[i].index
		currLabel = annotationArray[i-1].label

		for j in range(localLo,localHi):

			labelArray.append(currLabel)

	## for the last segment

	localLo = annotationArray[len(annotationArray)-1].index
	localHi = hi
	currLabel = annotationArray[len(annotationArray)-1].label

	for j in range(localLo, localHi):
		labelArray.append(currLabel)

	return labelArray				# point wise annotation array


def createCUDBAnnotation(annotationIndex,annotationArr,lenn):
	'''
	Create the annotation array for CUDB files
	
	Arguments:
		annotationIndex {list} -- list of indices
		annotationArr {list} -- list of labels
		lenn {int} -- length		
	
	Returns:
		[list] -- point wise annotation array
	'''

	li = []							# initialize

	for i in range(lenn):						
		li.append('notVF')					

	st=-1
	en=-1


	for i in range(len(annotationArr)):

		if(annotationArr[i]=='N'):				

			li[i]='NSR'

	for i in range(len(annotationArr)):

		if(annotationArr[i]=='['):

			st = annotationIndex[i]

		if(annotationArr[i]==']'):

			en = annotationIndex[i]

			for j in range(st,en+1):

				li[j]='VF'
			st = -1
			en = -1

	if(st!=-1):

		for j in range(st,lenn):
			li[j] = 'VF'

	return np.array(li)


def processMITMVADB(Te=5):
	'''
	Processes all mitMVA db file
	
	Keyword Arguments:
		Te {int} -- episode length (default: {5})
	'''


	Fs = 250			# sampling frequency

	print('Processing MITMVAdb files')

	for i in tqdm(range(400,700)):			# all mitMVA db files

		if (os.path.isfile("database/mitMVAdb/"+str(i) + ".dat")):			# file exists
			processMITMVADBFile(path='database/mitMVAdb/'+str(i),fileNo=i,Te=Te)


def processMITMVADBFile(path,fileNo,Te=5):
	'''
	Processes a mitMVA db file
	
	Arguments:
		path {str} -- path to file
		fileNo {int} -- number of file
	
	Keyword Arguments:
		Te {int} -- episode length (default: {5})
	'''


	signals, fields = wfdb.rdsamp(path)		# collect the signal and metadata
	Fs=fields['fs']							# sampling frequency 

	channel1Signal = []						# channel 1 signal
	channel2Signal = []						# channel 2 signal

	for i in signals:
											# separating the two channels
		channel1Signal.append(i[0])			
		channel2Signal.append(i[1])

	channel1Signal = np.array(channel1Signal)		# converting lists to numpy arrays
	channel2Signal = np.array(channel2Signal)

	annotation = wfdb.rdann(path, 'atr')			# collecting the annotation
	annotIndex = annotation.sample					# annotation indices
	annotSymbol = annotation.aux_note				# annotation symbols

	for i in range(len(annotSymbol)):

		annotSymbol[i] = annotSymbol[i].rstrip('\x00') # because the file contains \x00 

		if(annotSymbol[i]=='(N'):		# N = NSR
			annotSymbol[i]='(NSR'

		elif (annotSymbol[i] == '(VFIB'):	# VFIB = VF
			annotSymbol[i] = '(VF'

			# creating the annotation array
	annotationArr = createAnnotationArray(indexArray=annotIndex,labelArray=annotSymbol,hi=len(channel1Signal),NSRsymbol='(NSR') 

	nSamplesIn1Sec = Fs					# computing samples in one episode
	nSamplesInEpisode = Te * Fs	

	ecgSignals = []

	i=0									# episode counter

	while((i+nSamplesInEpisode)<len(channel1Signal)):			# loop through the whole signal

		j = i + nSamplesInEpisode

		VF = 0													# VF indices
		notVF = 0												# Not VF indices
		Noise =0												# Noise indices

		for k in range(i,j):

			if(annotationArr[k]=='(VF'):
				VF+=1
			else:						# anything other than VF
				notVF +=1

			if(annotationArr[k]=='(NOISE'):
				Noise += 1

		if(Noise*3<nSamplesInEpisode):						# noisy episode

																			# saving channel 1 signal
			ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='VF' if VF>notVF else 'NotVF',channel='Channel1',source='MITMVAdb',Fs=Fs)
			pickle.dump(ecgEpisode, open("Pickles/MITMVAdb/"+str(fileNo)+"E" + str(i // Fs) + "C1.p", "wb"))
																			
																			# saving channel 2 signal
			ecgEpisode = EcgSignal(signal=channel2Signal[i:j], annotation='VF' if VF > notVF else 'NotVF', channel='Channel2', source='MITMVAdb', Fs=Fs)
			pickle.dump(ecgEpisode, open("Pickles/MITMVAdb/"+str(fileNo)+"E" +  str(i // Fs) + "C2.p", "wb"))


		i += nSamplesIn1Sec								# sliding the window


def labelMITMVADBEpisodes(path,fileNo,Te=5):
	"""
	Labels the feature files from MITMVA db
	
	Arguments:
		path {str} -- path to file
		fileNo {int} -- number of file
	
	Keyword Arguments:
		Te {int} -- episode length (default: {5})
	"""


	signals, fields = wfdb.rdsamp(path)				# collect the signal and metadata
	Fs = fields['fs']								# sampling frequency 

	channel1Signal = []								# channel 1 signal
	channel2Signal = []								# channel 2 signal

	for i in signals:
													# separating the two channels
		channel1Signal.append(i[0])
		channel2Signal.append(i[1])

	channel1Signal = np.array(channel1Signal)		# converting lists to numpy arrays
	channel2Signal = np.array(channel2Signal)

	annotation = wfdb.rdann(path, 'atr')			# collecting the annotation
	annotIndex = annotation.sample					# annotation indices
	annotSymbol = annotation.aux_note				# annotation symbols

	for i in range(len(annotSymbol)):

		annotSymbol[i] = annotSymbol[i].rstrip('\x00')  # because the file contains \x00 

		if (annotSymbol[i] == '(N'):			# N = NSR
			annotSymbol[i] = '(NSR'

		elif (annotSymbol[i] == '(VFIB'):		# VFIB = VF
			annotSymbol[i] = '(VF'

						# creating the annotation array
	annotationArr = createAnnotationArray(indexArray=annotIndex, labelArray=annotSymbol, hi=len(channel1Signal), NSRsymbol='(NSR')

	nSamplesIn1Sec = Fs							# computing samples in one episode
	nSamplesInEpisode = Te * Fs

	ecgSignals = []

	i = 0										# episode counter

	while ((i + nSamplesInEpisode) < len(channel1Signal)):			# loop through the whole signal

		j = i + nSamplesInEpisode

		VF = 0														# VF indices
		notVF = 0													# Not VF indices
		Noise = 0													# Noise indices

		for k in range(i, j):

			if (annotationArr[k] == '(VF'):
				VF += 1
			else:								# anything other than VF
				notVF += 1

			if (annotationArr[k] == '(NOISE'):
				Noise += 1

		if(Noise * 3 < nSamplesInEpisode):				# noisy episode

			episodeId = str(i // Fs)

			enhancedAnnotation = createLabelsDict(VF=VF,notVF=notVF)

														# channel 1 signal
			if (os.path.isfile("Pickles/MITMVAdbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(1) + ".p")):

				dataa = pickle.load(open("Pickles/MITMVAdbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(1) + ".p",'rb'))

				dataa.label = enhancedAnnotation				

				pickle.dump(dataa,open("Pickles/MITMVAdbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(1) + ".p",'wb'))

														# channel 2 signal
			if (os.path.isfile("Pickles/MITMVAdbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(2) + ".p")):

				dataa = pickle.load(open("Pickles/MITMVAdbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(2) + ".p", 'rb'))

				dataa.label = enhancedAnnotation

				pickle.dump(dataa, open("Pickles/MITMVAdbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(2) + ".p", 'rb'))



		i += nSamplesIn1Sec				# sliding the window


def processCUDB(Te=5):
	'''
	Processes all cudb file
	
	Keyword Arguments:
		Te {int} -- episode length (default: {5})
	'''

	Fs = 250					# sampling frequency

	print('Processing CUdb files')


	for i in tqdm(range(40)):				# all mitMVA db files

		if (os.path.isfile('database/cudb/cu'+ (str(i) if i>9 else '0'+str(i))+".dat")):			# file exists
			
			processCUDBFile(path='database/cudb/cu'+ (str(i) if i>9 else '0'+str(i)) ,fileNo=i,Te=Te)


def processCUDBFile(path,fileNo,Te=5):
	'''
	Processes a cudb file
	
	Arguments:
		path {str} -- path to file
		fileNo {int} -- number of file
	
	Keyword Arguments:
		Te {int} -- episode length (default: {5})
	'''

	signals, fields = wfdb.rdsamp(path) 		# collect the signal and metadata
	Fs=fields['fs']								# sampling frequency 	

	channel1Signal = []							# channel 1 signal

	for i in signals:
												# separating the two channels
		channel1Signal.append(i[0])

	channel1Signal = np.array(channel1Signal)	# converting lists to numpy arrays

	annotation = wfdb.rdann(path, 'atr')		# collecting the annotation
	annotIndex = annotation.sample				# annotation indices
	annotSymbol = annotation.symbol				# annotation symbols

							# creating the annotation array
	annotationArray = createCUDBAnnotation(annotationIndex=annotIndex,annotationArr=annotSymbol,lenn=len(channel1Signal))

	nSamplesIn1Sec = Fs					# computing samples in one episode
	nSamplesInEpisode = Te * Fs

	ecgSignals = []

	i=0									# episode counter

	while((i+nSamplesInEpisode)<len(channel1Signal)):		# loop through the whole signal

		j = i + nSamplesInEpisode

		VF = 0												# VF indices
		NSR = 0												# NSR indices
		notVF = 0											# Not VF indices
		Noise = 0											# Noise indices


		for k in range(i,j):

			if(annotationArray[k]=='VF'):
				VF+=1
			elif(annotationArray[k]=='NSR'):	# unnecessary
				NSR += 1
			else:
				notVF +=1

		if(Noise*3<nSamplesInEpisode):						# noisy episode

			if(2*VF>=nSamplesInEpisode):
				ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='VF' if VF>notVF else 'NotVF',channel='Only_Channel',source='cudb',Fs=Fs)
				pickle.dump(ecgEpisode, open("Pickles/cudb/"+str(fileNo)+"E" + str(i // Fs) + "C1.p", "wb"))
			elif(2*NSR>=nSamplesInEpisode):
				ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='NSR',channel='Only_Channel',source='cudb',Fs=Fs)
				pickle.dump(ecgEpisode, open("Pickles/cudb/"+str(fileNo)+"E" + str(i // Fs) + "C1.p", "wb"))
			else:
				ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='NotVF',channel='Only_Channel',source='cudb',Fs=Fs)
				pickle.dump(ecgEpisode, open("Pickles/cudb/"+str(fileNo)+"E" + str(i // Fs) + "C1.p", "wb"))



		i += nSamplesIn1Sec								# sliding the window


def labelCUDBEpisodes(path,fileNo,Te=5):
	"""
	Labels the feature files from cudb
	
	Arguments:
		path {str} -- path to file
		fileNo {int} -- number of file
	
	Keyword Arguments:
		Te {int} -- episode length (default: {5})
	"""

	signals, fields = wfdb.rdsamp(path)				# collect the signal and metadata
	Fs = fields['fs']								# sampling frequency 
	

	channel1Signal = []								# channel 1 signal

	for i in signals:
		channel1Signal.append(i[0])

	channel1Signal = np.array(channel1Signal)		# converting lists to numpy arrays

	annotation = wfdb.rdann(path, 'atr')			# collecting the annotation
	annotIndex = annotation.sample					# annotation indices
	annotSymbol = annotation.symbol					# annotation symbols
							

							# creating the cudb annotation array
	annotationArray = createCUDBAnnotation(annotationIndex=annotIndex, annotationArr=annotSymbol, lenn=len(channel1Signal))


	nSamplesIn1Sec = Fs								# computing samples in one episode
	nSamplesInEpisode = Te * Fs

	ecgSignals = []

	i = 0											# episode counter

	while ((i + nSamplesInEpisode) < len(channel1Signal)):			# loop through the whole signal

		j = i + nSamplesInEpisode

		VF = 0														# VF indices
		NSR = 0														# NSR indices
		notVF = 0													# Not VF indices
		Noise = 0													# Noise indices

		for k in range(i, j):

			if (annotationArray[k] == 'VF'):
				VF += 1
			elif (annotationArray[k] == 'NSR'):
				NSR += 1										# unnecessary
			else:
				notVF += 1

		if (Noise * 3 < nSamplesInEpisode):						# noisy episode

			episodeId = str(i // Fs)

			enhancedAnnotation = createLabelsDict(VF=VF,notVF=NSR+notVF)


			if (os.path.isfile("Pickles/cudbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(1) + ".p")):

				dataa = pickle.load(open("Pickles/cudbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(1) + ".p",'rb'))

				dataa.label = enhancedAnnotation

				pickle.dump(dataa,open("Pickles/cudbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(1) + ".p",'wb'))



		i += nSamplesIn1Sec						# sliding the window


def downloadData():
	'''
		Downloads data from physionet
	'''


	try:							# creating necessary directories
		os.mkdir('database')
		os.mkdir('database/cudb')
		os.mkdir('database/mitMVAdb')
	except:
		pass 
	
	wfdb.dl_database('vfdb', dl_dir='database/mitMVAdb')	# download mitMVAdb
	wfdb.dl_database('cudb', dl_dir='database/cudb')		# download cudb


def processData(Te=5):
	"""
	Processes all data
	
	Keyword Arguments:
		Te {int} -- episode length (default: {5})
	"""

	
	try:							# creating the necessary directories
		os.mkdir('Pickles')
		os.mkdir('Pickles/cudb')
		os.mkdir('Pickles/MITMVAdb')
	except:
		pass

	processMITMVADB(Te)				# process MITMVA db files
	processCUDB(Te)					# process cudb files


def labelEpisodes(Te=5):
	"""
	Labels all the feature files
	
	Keyword Arguments:
		Te {int} -- episode length (default: {5})
	"""


									# label all MITMVA db files
	for i in tqdm(range(400,700)):

		if (os.path.isfile("database/mitMVAdb/"+str(i) + ".dat")):
			labelMITMVADBEpisodes(path='database/mitMVAdb/'+str(i),fileNo=i,Te=Te)

									# label all cudb files
	for i in tqdm(range(40)):

		if (os.path.isfile('database/cudb/cu'+ (str(i) if i>9 else '0'+str(i))+".dat")):
			labelCUDBEpisodes(path='database/cudb/cu'+ (str(i) if i>9 else '0'+str(i)) ,fileNo=i,Te=Te)




