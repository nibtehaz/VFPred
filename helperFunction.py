# Bismillahir Rahmanir Rahim
# Rabbi Zidni Ilma 

def cosineSimilarity(sig1,sig2):

	from scipy.spatial import distance
	import numpy as np 

	if(abs(np.sum(sig1)-0.0)<1e-5 or abs(np.sum(sig2)-0.0)<1e-5):
		return 0.0

	return  1 - distance.cosine(sig1, sig2)

