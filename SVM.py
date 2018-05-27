# Bismillahir Rahmanir Rahim
# Rabbi Zidni Ilma 

import numpy as np 
import pickle
from tqdm import tqdm
import os
from imblearn.over_sampling import SMOTE
from sklearn import svm

def createData(channel=1, percent=100, mitmvadb=True, cudb=True, saveFile='dataSet.p'):


	VF_features = []
	notVF_features = []

	'''
		Load MIT MVA DB
	'''
	if(mitmvadb):

		for i in tqdm(range(400, 700)):
			
			for j in (range(2100)):
								
				if (not os.path.isfile("Pickles/MITMVAdb/" + str(i) + "E" + str(j) + "C" + str(channel) + ".p")):
					continue

				dataa = pickle.load(open("Pickles/MITMVAdbFFT/F" + str(i) + "E" + str(j) + "C" + str(channel) + ".p", "rb"))

				features = []

				normalization = (np.sum(np.power(np.abs(dataa.Signal_FFT), 2)) * np.sum(np.power(np.abs(dataa.IMF1_FFT), 2))) ** 0.5

				for k in range(len(dataa.Signal_FFT)):
					features.append(np.abs(dataa.Signal_FFT[k]) * np.abs(dataa.IMF1_FFT[k]) / normalization)

				normalization = (np.sum(np.power(np.abs(dataa.Signal_FFT), 2)) * np.sum(np.power(np.abs(dataa.R_FFT), 2))) ** 0.5

				for k in range(len(dataa.Signal_FFT)):
					features.append(np.abs(dataa.Signal_FFT[k]) * np.abs(dataa.R_FFT[k]) / normalization)

				if (dataa.label[(percent//10)-1] == 1):

					VF_features.append(np.array(features))

				else:

					notVF_features.append(np.array(features))

	

	'''
		Load CUDB
	'''

	if(cudb):

		for i in tqdm(range(40)):
			
			for j in (range(550)):
								

				if (not os.path.isfile("Pickles/cudbFFT/F" + str(i) + "E" + str(j) + "C" + str(1) + ".p")):
					continue

				dataa = pickle.load(open("Pickles/cudbFFT/F" + str(i) + "E" + str(j) + "C" + str(1) + ".p", "rb"))

				features = []

				normalization = (np.sum(np.power(np.abs(dataa.Signal_FFT), 2)) * np.sum(np.power(np.abs(dataa.IMF1_FFT), 2))) ** 0.5

				for k in range(len(dataa.Signal_FFT)):
					features.append(np.abs(dataa.Signal_FFT[k]) * np.abs(dataa.IMF1_FFT[k]) / normalization)

				normalization = (np.sum(np.power(np.abs(dataa.Signal_FFT), 2)) * np.sum(np.power(np.abs(dataa.R_FFT), 2))) ** 0.5

				for k in range(len(dataa.Signal_FFT)):
					features.append(np.abs(dataa.Signal_FFT[k]) * np.abs(dataa.R_FFT[k]) / normalization)

				if (dataa.label[(percent // 10) - 1] == 1):

					VF_features.append(np.array(features))


				else:

					notVF_features.append(np.array(features))

	pickle.dump((VF_features, notVF_features), open(saveFile, "wb"))

def loadData(vfCnt=1800,notVfCnt=2400,saveFile='dataSet.p'):

	dataa = pickle.load(open(saveFile, "rb"))

	VF_features = dataa[0]
	notVF_features = dataa[1]

	np.random.shuffle(VF_features)
	np.random.shuffle(notVF_features)

	Train = []
	Test = []

	X_Train = []
	Y_Train = []
	X_Test = []
	Y_Test=[]


	for i in range(vfCnt):

		Train.append((VF_features[i],1))

	for i in range(vfCnt,len(VF_features)):

		Test.append((VF_features[i], 1))

	for i in range(notVfCnt):
		Train.append((notVF_features[i], 0))

	for i in range(notVfCnt, len(notVF_features)):
		Test.append((notVF_features[i], 0))

	np.random.shuffle(Train)
	np.random.shuffle(Test)

	for i in range(len(Train)):

		X_Train.append(Train[i][0])
		Y_Train.append(Train[i][1])

	for i in range(len(Test)):
		X_Test.append(Test[i][0])
		Y_Test.append(Test[i][1])

	return (X_Train, Y_Train, X_Test, Y_Test)

def upsampleSMOTE(loadFile='dataSet.p',saveFile='smoteData.p'):

	dataa = pickle.load(open(loadFile, "rb"))

	VF_features = dataa[0]
	notVF_features = dataa[1]

	np.random.shuffle(VF_features)
	np.random.shuffle(notVF_features)

	X = []
	Y = []

	for i in VF_features:

		X.append(i)
		Y.append(1)

	for i in notVF_features:

		X.append(i)
		Y.append(0)

	sm = SMOTE(kind='regular')

	Xup , Yup = sm.fit_sample(X,Y)

	VF_features = []
	notVF_features = []

	for i in range(len(Xup)):

		if(Yup[i]==1):
			VF_features.append(Xup[i])
		else:
			notVF_features.append(Xup[i])

	pickle.dump((VF_features, notVF_features), open(saveFile, "wb"))

def featureSelection(X,percentage=80):

	featureRanking = pickle.load(open('featureRanking.p','rb'))
	#print(featureRanking)

	lenn = (len(X[0])*percentage)//100

	trimmedX = []

	for i in X :

		li = []

		for j in range(lenn):

			li.append(i[featureRanking[j]])

		trimmedX.append(li)

	return np.array(trimmedX)

def evaluate(clf,X,Y,returning=False):

	trueVF = 0
	falseVF = 0
	trueNotVF = 0
	falseNotVF = 0

	for i in range(len(X)):

		#print(i,end=' ',flush=True)

		yP = clf.predict([X[i]])
		#print(yP, Y[i])

		if (yP[0] < 0.5):

			if (Y[i] == 0):

				trueNotVF += 1

			else:

				falseNotVF += 1

		else:

			if (Y[i] == 1):

				trueVF += 1

			else:

				falseVF += 1

	if returning:
		return [str(trueNotVF * 100.0 / (trueNotVF + falseVF)), str(trueVF * 100.0 / (trueVF + falseNotVF)), str((trueVF + trueNotVF) * 100.0 / (trueVF + falseVF + trueNotVF + falseNotVF))]

	print('trueNotVF : ' + str(trueNotVF))
	print('trueVF : ' + str(trueVF))
	print('falseNotVF : ' + str(falseNotVF))
	print('falseVF : ' + str(falseVF))
	print('Specificity : '+str(trueNotVF*100.0/(trueNotVF+falseVF)))
	print('Sensitivity : ' + str(trueVF * 100.0 / (trueVF + falseNotVF)))
	print('Accuracy : ' + str((trueVF + trueNotVF) * 100.0 / (trueVF + falseVF + trueNotVF + falseNotVF)))



def kFoldCrossValidation(gamma=45,C=100,file='smoteData.p',featurePercent=50):

	dataa = pickle.load(open(file, "rb"))

	VF_features = dataa[0]
	notVF_features = dataa[1]

	VFcnt = len(VF_features)//10
	notVFcnt = len(VF_features) // 10

	VFmark = []
	notVFmark = []

	for k in range(10):

		VFmark.append( [ k*VFcnt , (k+1)*VFcnt ] )
		notVFmark.append([k * notVFcnt, (k + 1) * notVFcnt])

	VFmark[len(VFmark)-1][1]=len(VF_features)
	notVFmark[len(notVFmark)-1][1] = len(notVF_features)

	for k in range(10):

		X_Train = []
		Y_Train = []
		X_Test = []
		Y_Test = []

		for i in range(10):

			if(i==k):

				for j in range(VFmark[i][0], VFmark[i][1]):

					X_Test.append(VF_features[j])
					Y_Test.append(1)

				for j in range(notVFmark[i][0], notVFmark[i][1]):

					X_Test.append(notVF_features[j])
					Y_Test.append(0)

			else:

				for j in range(VFmark[i][0],VFmark[i][1]):

					X_Train.append(VF_features[j])
					Y_Train.append(1)

				for j in range(notVFmark[i][0], notVFmark[i][1]):

					X_Train.append(notVF_features[j])
					Y_Train.append(0)
		

		print('**********************')
		print(k+1)
		clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)

		X_Train = featureSelection(X=X_Train, percentage=featurePercent)
		X_Test = featureSelection(X=X_Test, percentage=featurePercent)
		clf.fit(X_Train, Y_Train)


		evaluate(clf, X_Test, Y_Test, returning=False)

		print('**********************')
		print()



def main():
	
	np.random.seed(3)
	upsampleSMOTE()


if __name__ == '__main__':
	main()