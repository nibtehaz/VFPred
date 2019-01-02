'''
Codes for machine learning 
'''

import matplotlib.pyplot as plt
from helper_function import cosineSimilarity
import numpy as np
from data_structures import Features
from scipy.fftpack import ifft
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
import pickle
from sklearn import svm
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def createData(channel, percent=100 , mitmvadb=True,cudb=True, saveFile='dataSetType3.p'):
	"""
	Create the data file that would used for machine learning 
	
	Arguments:
		channel {int} -- which channel to use
	
	Keyword Arguments:
		percent {int} -- percent of VF samples needed to be labeled as VF (default: {100})
		mitmvadb {bool} -- include mitMVA db dataset ? (default: {True})
		cudb {bool} -- include cudb dataset ? (default: {True})
		saveFile {str} -- name of save file (default: {'dataSetType3.p'})
	"""

	VF_features = []
	notVF_features = []

	'''
		Load MIT MVA DB
	'''
	if(mitmvadb):

		for i in range(400, 700):		# all mitmva db files

			for j in range(2100):					# all mitmva db episodes				

				if (not os.path.isfile("Pickles/MITMVAdb/" + str(i) + "E" + str(j) + "C" + str(channel) + ".p")):
								# no file
					continue		

								# load features 
				dataa = pickle.load(open("Pickles/MITMVAdbFFT/F" + str(i) + "E" + str(j) + "C" + str(channel) + ".p", "rb"))
						
				features = []
															# normalization factor
				normalization = (np.sum(np.power(np.abs(dataa.Signal_FFT), 2)) * np.sum(np.power(np.abs(dataa.IMF1_FFT), 2))) ** 0.5		

				for k in range(len(dataa.Signal_FFT)):
					features.append(np.abs(dataa.Signal_FFT[k]) * np.abs(dataa.IMF1_FFT[k]) / normalization)
															
															# normalization factor
				normalization = (np.sum(np.power(np.abs(dataa.Signal_FFT), 2)) * np.sum(np.power(np.abs(dataa.R_FFT), 2))) ** 0.5

				for k in range(len(dataa.Signal_FFT)):
					features.append(np.abs(dataa.Signal_FFT[k]) * np.abs(dataa.R_FFT[k]) / normalization)


				if (dataa.label[(percent//10)-1] == 1):			# label VF or not VF

					VF_features.append(np.array(features))

				else:

					notVF_features.append(np.array(features))


	'''
		Load CUDB
	'''

	if(cudb):

		for i in range(0, 40):			# all cudb files

			for j in range(550):			# all cudb episodes
				

				if (not os.path.isfile("Pickles/cudbFFT/F" + str(i) + "E" + str(j) + "C" + str(1) + ".p")):
							# no file
					continue

							# load features 
				dataa = pickle.load(open("Pickles/cudbFFT/F" + str(i) + "E" + str(j) + "C" + str(1) + ".p", "rb"))

				features = []

											# normalization factor
				normalization = (np.sum(np.power(np.abs(dataa.Signal_FFT), 2)) * np.sum(np.power(np.abs(dataa.IMF1_FFT), 2))) ** 0.5

				for k in range(len(dataa.Signal_FFT)):
					features.append(np.abs(dataa.Signal_FFT[k]) * np.abs(dataa.IMF1_FFT[k]) / normalization)

											# normalization factor
				normalization = (np.sum(np.power(np.abs(dataa.Signal_FFT), 2)) * np.sum(np.power(np.abs(dataa.R_FFT), 2))) ** 0.5

				for k in range(len(dataa.Signal_FFT)):
					features.append(np.abs(dataa.Signal_FFT[k]) * np.abs(dataa.R_FFT[k]) / normalization)

				if (dataa.label[(percent // 10) - 1] == 1):			# label VF or not VF

					VF_features.append(np.array(features))


				else:

					notVF_features.append(np.array(features))

	
																	# save the features as a pickle file
	pickle.dump((VF_features, notVF_features), open(saveFile, "wb"))


def loadData(vfCnt=1800,notVfCnt=2400,file='dataSetType3.p'):
	"""
	Loads data
	
	Keyword Arguments:
		vfCnt {int} -- number of vf samples (default: {1800})
		notVfCnt {int} -- number of not vf samples (default: {2400})
		file {str} -- name of file (default: {'dataSetType3.p'})
	
	Returns:
		[tuple] -- tuple containing (X_Train, Y_Train, X_Test, Y_Test)
	"""

	dataa = pickle.load(open(file, "rb"))		# load the features

	VF_features = dataa[0]						
	notVF_features = dataa[1]
													# random shuffling 
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

	np.random.shuffle(Train)			# random shuffle
	np.random.shuffle(Test)

	for i in range(len(Train)):

		X_Train.append(Train[i][0])
		Y_Train.append(Train[i][1])

	for i in range(len(Test)):
		X_Test.append(Test[i][0])
		Y_Test.append(Test[i][1])

	return (X_Train, Y_Train, X_Test, Y_Test)


def upsampleSMOTE(loadFile='dataSetType3.p',saveFile='smoteData.p'):
	"""
	Generate synthetic data using smote
	
	Keyword Arguments:
		loadFile {str} -- file to load (default: {'dataSetType3.p'})
		saveFile {str} -- file to save (default: {'smoteData.p'})
	"""

	dataa = pickle.load(open(loadFile, "rb"))
										# loading the features
	VF_features = dataa[0]
	notVF_features = dataa[1]
											# shuffling
	np.random.shuffle(VF_features)
	np.random.shuffle(notVF_features)

	X = []
	Y = []

	for i in VF_features:				# adding the VF features

		X.append(i)
		Y.append(1)

	for i in notVF_features:			# adding the not VF features

		X.append(i)
		Y.append(0)

	sm = SMOTE(kind='regular')			# smote object

	Xup , Yup = sm.fit_sample(X,Y)		# generate synthetic data

	VF_features = []
	notVF_features = []

	for i in range(len(Xup)):

		if(Yup[i]==1):						# VF feature
			VF_features.append(Xup[i])	
		else:								# not VF feature
			notVF_features.append(Xup[i])
																		# saving the SMOTE'd data
	pickle.dump((VF_features, notVF_features), open(saveFile, "wb"))


def featureRanking(X,Y):
	"""
	Ranks the features using a Random Forest
	
	Arguments:
		X {array} -- features
		Y {array} -- labels
	"""


	forest = RandomForestClassifier(n_estimators=750,random_state=3,verbose=2)
	forest.fit(X, Y)
	pickle.dump(forest,open('randomForest750.p','wb'))

	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
	indices = np.argsort(importances)[::-1]

	pickle.dump(indices, open('featureRanking.p', 'wb'))

	
def featureSelection(X,percentage=24):
	"""
	Select the top percentage of features
	
	Arguments:
		X {2D list} -- all features of all samples
	
	Keyword Arguments:
		percentage {int} -- percentage of top features (default: {24})
	
	Returns:
		[numpy array] -- selected top features
	"""


	featureRanking = pickle.load(open('featureRanking.p','rb'))		# load the feature ranking / order 

	lenn = (len(X[0])*percentage)//100		# number of features to consider

	trimmedX = np.zeros((len(X), lenn))		# initializing 

	for i in range(len(X)) :

		for j in range(lenn):			

			trimmedX[i][j] = X[i][featureRanking[j]]
	
	X = None			# garbage collection

	return trimmedX


def evaluate(clf,X,Y,returning=False):
	"""
	Evaluate the algorithm
	
	Arguments:
		clf {skelarn svm model} -- trained SVM model
		X {numpy array} -- features
		Y {numpy array} -- labels
	
	Keyword Arguments:
		returning {bool} -- return the results ? (default: {False})
	
	Returns:
		null or str -- results
	"""

	trueVF = 0			# TP
	falseVF = 0			# FP
	trueNotVF = 0		# TN
	falseNotVF = 0		# FN

	for i in range(len(X)):

		yP = clf.predict([X[i]])

		if (yP[0] < 0.5):

			if (Y[i] == 0):

				trueNotVF += 1			# TN

			else:

				falseNotVF += 1			# FN

		else:

			if (Y[i] == 1):

				trueVF += 1				# TP

			else:

				falseVF += 1			# FP

	if returning:			# returns the results

		return [str(trueNotVF * 100.0 / (trueNotVF + falseVF)), str(trueVF * 100.0 / (trueVF + falseNotVF)), str((trueVF + trueNotVF) * 100.0 / (trueVF + falseVF + trueNotVF + falseNotVF))]

							# or just print it 

	print('trueNotVF : ' + str(trueNotVF))
	print('trueVF : ' + str(trueVF))
	print('falseNotVF : ' + str(falseNotVF))
	print('falseVF : ' + str(falseVF))
	print('Specificity : '+str(trueNotVF*100.0/(trueNotVF+falseVF)))
	print('Sensitivity : ' + str(trueVF * 100.0 / (trueVF + falseNotVF)))
	print('Accuracy : ' + str((trueVF + trueNotVF) * 100.0 / (trueVF + falseVF + trueNotVF + falseNotVF)))


def svmParameterTuning(file='dataSetType3.p',vfCnt=3000,notvfCnt=5000):
	"""
	Grid search for svm paramter tuning
	
	Keyword Arguments:
		file {str} -- name of data file (default: {'dataSetType3.p'})
		vfCnt {int} -- number of vf samples (default: {3000})
		notvfCnt {int} -- number of not vf samples (default: {5000})		
	"""


	gammas = [5,10,15,20,25,30,35,40,45,50,55,60]		# list of gamma values
	Cs = [100,10,1,0.1]									# list of C values

	(X_Train, Y_Train, X_Test, Y_Test) = loadData(vfCnt=vfCnt, notVfCnt=notvfCnt, file=file)		
														# get train test split

							# exhaustive grid search
	for gamma in gammas:
		for C in Cs:
			clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
			clf.fit(X_Train, Y_Train)
			pp=evaluate(clf,X_Test,Y_Test,returning=True)
			print('"'+str(gamma)+' , '+str(C)+' : ( '+str(pp[0])+' , '+str(pp[1])+' , '+str(pp[2])+'"')


def kFoldCrossValidation(gamma=45,C=100,file='smoteData.p',featurePercent=24):
	'''
	Perform K fold cross validation
	
	Keyword Arguments:
		gamma {int} -- parameter for svm (default: {45})
		C {int} -- parameter for svm (default: {100})
		file {str} -- data file (default: {'smoteData.p'})
		featurePercent {int} -- percentage of features to be used (default: {24})
	'''

	dataa = pickle.load(open(file, "rb"))

	X = dataa[0][:]		# VF features
	Y = []				# labels
	

	for i in range(len(X)):		# adding the VF labels
		
		Y.append(1)

	for i in range(len(dataa[1])):	# adding the not VF features and labels

		X.append(dataa[1][i])
		Y.append(0)

	dataa = None		# garbage collection

	X = featureSelection(X=X, percentage=featurePercent)	# select the predefined number of top features

						# converting lists to numpy arrays
	X = np.array(X)		
	Y = np.array(Y)

	kf = KFold(n_splits=10,shuffle=True,random_state=3)
	kf.get_n_splits(X,Y)

	k = 1

	for train_index, test_index in kf.split(X):

		X_Train, X_Test = X[train_index], X[test_index]
		Y_Train, Y_Test = Y[train_index], Y[test_index]
	
		print('**********************')
		print(k)
		clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)

		clf.fit(X_Train, Y_Train)

		print('Training')
		evaluate(clf, X_Train, Y_Train, returning=False)
		print('Testing')
		evaluate(clf, X_Test, Y_Test, returning=False)

		print('**********************')
		print()
		k+= 1 


def stratifiedkFoldCrossValidation(gamma=45,C=100,file='smoteData.p',featurePercent=24):
	'''
	Perform Stratified K fold cross validation
	
	Keyword Arguments:
		gamma {int} -- parameter for svm (default: {45})
		C {int} -- parameter for svm (default: {100})
		file {str} -- data file (default: {'smoteData.p'})
		featurePercent {int} -- percentage of features to be used (default: {24})
	'''

	dataa = pickle.load(open(file, "rb"))

	X = dataa[0][:]		# VF features
	Y = []				# labels
	

	for i in range(len(X)):		# adding the VF labels
		
		Y.append(1)

	for i in range(len(dataa[1])):	# adding the not VF features and labels

		X.append(dataa[1][i])
		Y.append(0)

	dataa = None		# garbage collection

	X = featureSelection(X=X, percentage=featurePercent)	# select the predefined number of top features

						# converting lists to numpy arrays
	X = np.array(X)		
	Y = np.array(Y)

	skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=3)
	skf.get_n_splits(X,Y)

	k = 1

	for train_index, test_index in skf.split(X,Y):

		X_Train, X_Test = X[train_index], X[test_index]
		Y_Train, Y_Test = Y[train_index], Y[test_index]
	
		print('**********************')
		print(k)
		clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)

		clf.fit(X_Train, Y_Train)

		print('Training')
		evaluate(clf, X_Train, Y_Train, returning=False)
		print('Testing')
		evaluate(clf, X_Test, Y_Test, returning=False)

		print('**********************')
		print()
		k+= 1 


if __name__=='__main__':

	np.random.seed(3)

	createData(1,100,saveFile='dataSetType3.p')
	
	svmParameterTuning(file='dataSetType3.p')
	
	upsampleSMOTE()
	
	kFoldCrossValidation(featurePercent=24)
	#stratifiedkFoldCrossValidation(featurePercent=24)
