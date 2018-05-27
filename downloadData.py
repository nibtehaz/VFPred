# Bismillahir Rahmanir Rahim
# Rabbi Zidni Ilma

import wfdb
import os 
import pickle
import numpy as np
from tqdm import tqdm
from dataStructures import EcgSignal, Annotation

def createLabelsDict(VF,notVF):

	# Return a numpy array of lenth 10
	# 1st element is 1 if atleast 10% of the samples are VF
	# 2nd element is 1 if atleast 20% of the samples are VF
	# ...
	# ...
	# ...
	# 10th element is 1 if atleast 100% of the samples are VF

	li = []

	tot = VF + notVF

	for i in range(1,11):

		if(VF>=((tot*i)//10)):

			li.append(1)

		else:
			li.append(0)

	return np.array(li)

def createAnnotationArray(indexArray,labelArray,hi,NSRsymbol):

	annotations = []

	for i in range(len(indexArray)):

		annotations.append(Annotation(index=indexArray[i],label=labelArray[i]))

	distributedAnnotations = createDistributedAnnotations(annotationArray=annotations,hi=hi,NSRsymbol=NSRsymbol)

	return distributedAnnotations

def createDistributedAnnotations(annotationArray,hi,NSRsymbol):

	labelArray=[]

	localLo = 0
	localHi = annotationArray[0].index
	currLabel = NSRsymbol

	## We are assuming the first unannotated part to be NSR

	for i in range(localLo,localHi):

		labelArray.append(currLabel)


	## now for the other actual annotated segments

	for i in range(1,len(annotationArray)):

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

	return labelArray

def processMITMVADB(Te=5):

	Fs = 250

	print('Processing MITMVAdb files')

	for i in tqdm(range(400,700)):

		if (os.path.isfile("database/mitMVAdb/"+str(i) + ".dat")):
			processMITMVADBFile(path='database/mitMVAdb/'+str(i),fileNo=i,Te=Te)

def processMITMVADBFile(path,fileNo,Te=5):

	signals, fields = wfdb.rdsamp(path)
	Fs=fields['fs']
	#print(fields)

	channel1Signal = []
	channel2Signal = []

	for i in signals:

		channel1Signal.append(i[0])
		channel2Signal.append(i[1])

	channel1Signal = np.array(channel1Signal)
	channel2Signal = np.array(channel2Signal)

	annotation = wfdb.rdann(path, 'atr')
	annotIndex = annotation.sample
	annotSymbol = annotation.aux_note

	for i in range(len(annotSymbol)):

		annotSymbol[i] = annotSymbol[i].rstrip('\x00') ## because the file contains so

		if(annotSymbol[i]=='(N'):
			annotSymbol[i]='(NSR'

		elif (annotSymbol[i] == '(VFIB'):
			annotSymbol[i] = '(VF'


	#for i in range(len(annotSymbol)):

		#print(annotIndex[i],annotSymbol[i])

	annotationArr = createAnnotationArray(indexArray=annotIndex,labelArray=annotSymbol,hi=len(channel1Signal),NSRsymbol='(NSR')

	nSamplesIn1Sec = Fs
	nSamplesInEpisode = Te * Fs

	ecgSignals = []

	i=0

	while((i+nSamplesInEpisode)<len(channel1Signal)):

		j = i + nSamplesInEpisode

		VF = 0
		notVF = 0
		Noise =0


		for k in range(i,j):

			if(annotationArr[k]=='(VF'):
				VF+=1
			else:
				notVF +=1

			if(annotationArr[k]=='(NOISE'):
				Noise += 1

		if(Noise*3<nSamplesInEpisode):

			ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='VF' if VF>notVF else 'NotVF',channel='Channel1',source='MITMVAdb',Fs=Fs)
			pickle.dump(ecgEpisode, open("Pickles/MITMVAdb/"+str(fileNo)+"E" + str(i // Fs) + "C1.p", "wb"))

			ecgEpisode = EcgSignal(signal=channel2Signal[i:j], annotation='VF' if VF > notVF else 'NotVF', channel='Channel2', source='MITMVAdb', Fs=Fs)
			pickle.dump(ecgEpisode, open("Pickles/MITMVAdb/"+str(fileNo)+"E" +  str(i // Fs) + "C2.p", "wb"))


		#print(fileNo ,i / Fs)

		i += nSamplesIn1Sec

def labelMITMVADBEpisodes(path,fileNo,Te=5):

	signals, fields = wfdb.rdsamp(path)
	Fs = fields['fs']
	#print(fields)

	channel1Signal = []
	channel2Signal = []

	for i in signals:
		channel1Signal.append(i[0])
		channel2Signal.append(i[1])

	channel1Signal = np.array(channel1Signal)
	channel2Signal = np.array(channel2Signal)

	annotation = wfdb.rdann(path, 'atr')
	annotIndex = annotation.sample
	annotSymbol = annotation.aux_note

	for i in range(len(annotSymbol)):

		annotSymbol[i] = annotSymbol[i].rstrip('\x00')  ## because the file contains so

		if (annotSymbol[i] == '(N'):
			annotSymbol[i] = '(NSR'

		elif (annotSymbol[i] == '(VFIB'):
			annotSymbol[i] = '(VF'

	#for i in range(len(annotSymbol)):
		#print(annotIndex[i], annotSymbol[i])

	annotationArr = createAnnotationArray(indexArray=annotIndex, labelArray=annotSymbol, hi=len(channel1Signal), NSRsymbol='(NSR')

	nSamplesIn1Sec = Fs
	nSamplesInEpisode = Te * Fs

	ecgSignals = []

	i = 0

	while ((i + nSamplesInEpisode) < len(channel1Signal)):

		j = i + nSamplesInEpisode

		VF = 0
		notVF = 0
		Noise = 0

		for k in range(i, j):

			if (annotationArr[k] == '(VF'):
				VF += 1
			else:
				notVF += 1

			if (annotationArr[k] == '(NOISE'):
				Noise += 1

		if(Noise * 3 < nSamplesInEpisode):

			episodeId = str(i // Fs)

			enhancedAnnotation = createLabelsDict(VF=VF,notVF=notVF)

			if (os.path.isfile("Pickles/MITMVAdbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(1) + ".p")):

				dataa = pickle.load(open("Pickles/MITMVAdbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(1) + ".p",'rb'))

				dataa.label = enhancedAnnotation

				#print(dataa.label)

				pickle.dump(dataa,open("Pickles/MITMVAdbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(1) + ".p",'wb'))

			if (os.path.isfile("Pickles/MITMVAdbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(2) + ".p")):

				dataa = pickle.load(open("Pickles/MITMVAdbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(2) + ".p", 'rb'))

				dataa.label = enhancedAnnotation

				pickle.dump(dataa, open("Pickles/MITMVAdbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(2) + ".p", 'rb'))


		#print(fileNo, i / Fs)

		i += nSamplesIn1Sec

def processCUDB(Te=5):

	Fs = 250 # sampling rate of CUDB files

	print('Processing CUdb files')

	# instead of using os.walk we did this for other convenience and laziness							
	for i in tqdm(range(40)):

		if (os.path.isfile('database/cudb/cu'+ (str(i) if i>9 else '0'+str(i))+".dat")):
			
			processCUDBFile(path='database/cudb/cu'+ (str(i) if i>9 else '0'+str(i)) ,fileNo=i,Te=Te)

def processCUDBFile(path,fileNo,Te=5):

	signals, fields = wfdb.rdsamp(path) 
	Fs=fields['fs']
	#print(fields)

	channel1Signal = []

	for i in signals:

		channel1Signal.append(i[0])

	channel1Signal = np.array(channel1Signal)

	annotation = wfdb.rdann(path, 'atr')
	annotIndex = annotation.sample
	annotSymbol = annotation.symbol

	annotationArray = createCUDBAnnotation(annotationIndex=annotIndex,annotationArr=annotSymbol,lenn=len(channel1Signal))

	#print(annotationArray)

	nSamplesIn1Sec = Fs
	nSamplesInEpisode = Te * Fs

	ecgSignals = []

	i=0

	while((i+nSamplesInEpisode)<len(channel1Signal)):

		j = i + nSamplesInEpisode

		VF = 0
		NSR = 0
		notVF = 0
		Noise =0


		for k in range(i,j):

			if(annotationArray[k]=='VF'):
				VF+=1
			elif(annotationArray[k]=='NSR'):
				NSR += 1
			else:
				notVF +=1

		if(Noise*3<nSamplesInEpisode):

			if(2*VF>=nSamplesInEpisode):
				ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='VF' if VF>notVF else 'NotVF',channel='Only_Channel',source='cudb',Fs=Fs)
				pickle.dump(ecgEpisode, open("Pickles/cudb/"+str(fileNo)+"E" + str(i // Fs) + "C1.p", "wb"))
			elif(2*NSR>=nSamplesInEpisode):
				ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='NSR',channel='Only_Channel',source='cudb',Fs=Fs)
				pickle.dump(ecgEpisode, open("Pickles/cudb/"+str(fileNo)+"E" + str(i // Fs) + "C1.p", "wb"))
			else:
				ecgEpisode = EcgSignal(signal=channel1Signal[i:j],annotation='NotVF',channel='Only_Channel',source='cudb',Fs=Fs)
				pickle.dump(ecgEpisode, open("Pickles/cudb/"+str(fileNo)+"E" + str(i // Fs) + "C1.p", "wb"))


		#print(fileNo, i / Fs)

		i += nSamplesIn1Sec

def createCUDBAnnotation(annotationIndex,annotationArr,lenn):

	li = []

	for i in range(lenn):
		li.append('notVF')

	st=-1
	en=-1

	#print(annotationArr)

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

def labelCUDBEpisodes(path,fileNo,Te=5):

	signals, fields = wfdb.rdsamp(path)
	Fs = fields['fs']
	#print(fields)

	channel1Signal = []

	for i in signals:
		channel1Signal.append(i[0])

	channel1Signal = np.array(channel1Signal)

	annotation = wfdb.rdann(path, 'atr')
	annotIndex = annotation.sample
	annotSymbol = annotation.symbol

	annotationArray = createCUDBAnnotation(annotationIndex=annotIndex, annotationArr=annotSymbol, lenn=len(channel1Signal))

	# print(annotationArray)


	nSamplesIn1Sec = Fs
	nSamplesInEpisode = Te * Fs

	ecgSignals = []

	i = 0

	while ((i + nSamplesInEpisode) < len(channel1Signal)):

		j = i + nSamplesInEpisode

		VF = 0
		NSR = 0
		notVF = 0
		Noise = 0

		for k in range(i, j):

			if (annotationArray[k] == 'VF'):
				VF += 1
			elif (annotationArray[k] == 'NSR'):
				NSR += 1
			else:
				notVF += 1

		if (Noise * 3 < nSamplesInEpisode):

			episodeId = str(i // Fs)

			enhancedAnnotation = createLabelsDict(VF=VF,notVF=notVF)


			if (os.path.isfile("Pickles/cudbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(1) + ".p")):

				dataa = pickle.load(open("Pickles/cudbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(1) + ".p",'rb'))

				dataa.label = enhancedAnnotation

				pickle.dump(dataa,open("Pickles/cudbFFT/F" + str(fileNo) + "E" + episodeId + "C" + str(1) + ".p",'wb'))


		#print(fileNo, i / Fs)

		i += nSamplesIn1Sec

def downloadData():

	try:
		os.mkdir('database')
		os.mkdir('database/cudb')
		os.mkdir('database/mitMVAdb')
	except:
		pass 
	
	wfdb.dl_database('vfdb', dl_dir='database/mitMVAdb')
	wfdb.dl_database('cudb', dl_dir='database/cudb')


def processData(Te=5):
	
	try:
		os.mkdir('Pickles')
		os.mkdir('Pickles/cudb')
		os.mkdir('Pickles/MITMVAdb')
	except:
		pass

	processMITMVADB(Te)
	processCUDB(Te)

def labelEpisodes(Te=5):

	for i in tqdm(range(400,700)):

		if (os.path.isfile("database/mitMVAdb/"+str(i) + ".dat")):
			labelMITMVADBEpisodes(path='database/mitMVAdb/'+str(i),fileNo=i,Te=Te)
	
	for i in tqdm(range(40)):

		if (os.path.isfile('database/cudb/cu'+ (str(i) if i>9 else '0'+str(i))+".dat")):
			labelCUDBEpisodes(path='database/cudb/cu'+ (str(i) if i>9 else '0'+str(i)) ,fileNo=i,Te=Te)




