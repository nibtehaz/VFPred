# VFPred

This repository contains the codes of the following paper

### VFPred: A Fusion of Signal Processing and Machine Learning techniques in Detecting Ventricular Fibrillation from ECG Signals.
#### Nabil Ibtehaz, M. Saifur Rahman, M. Sohel Rahman

</br>

Paper
[Preprint](https://arxiv.org/abs/1807.02684)
Project Website  
Project Blogpost 

## Abstract
Ventricular Fibrillation (VF), one of the most dangerous arrhythmias, is responsible for sudden cardiac arrests. Thus, various algorithms have been developed to predict VF from Electrocardiogram (ECG), which is a binary classification problem. In the literature, we find a number of algorithms based on signal processing, where, after some robust mathematical operations the decision is given based on a predefined threshold over a single value. On the other hand, some machine learning based algorithms are also reported in the literature; however, these algorithms merely combine some parameters and make a prediction using those as features. Both the approaches have their perks and pitfalls; thus our motivation was to coalesce them to get the best out of the both worlds. Hence we have developed, VFPred that, in addition to employing a signal processing pipeline, namely, Empirical Mode Decomposition and Discrete Time Fourier Transform for useful feature extraction, uses a Support Vector Machine for efficient classification. VFPred turns out to be a robust algorithm as it is able to successfully segregate the two classes with equal confidence (Sensitivity = 99.99%, Specificity = 98.40%) even from a short signal of 5 seconds long, whereas existing works though requires longer signals, flourishes in one but fails in the other.


## Citation

If you use VFPred in your work, please cite the following paper