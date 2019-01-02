"""
Helper functions
"""

from scipy.spatial import distance
import numpy as np 


def cosineSimilarity(sig1,sig2):
	'''
	Computes cosine similarity of two signals
	
	Arguments:
		sig1 {list} -- signal 1
		sig2 {list} -- signal 2
	
	Returns:
		[float] -- cosine similarity value
	'''

	if(abs(np.sum(sig1)-0.0)<1e-5 or abs(np.sum(sig2)-0.0)<1e-5):	# special case, if one of them equals zero
		return 0.0

	return  1 - distance.cosine(sig1, sig2)		# making most similar -> 1 , least similar -> 0

