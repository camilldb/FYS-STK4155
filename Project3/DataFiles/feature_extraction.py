import numpy as np
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
import pandas as pd

# Class for feature extraction and prediction using this features:
class FeatureExtraction:
    def __init__(self, trial):
        self.trial = trial
        self.n_channels = trial.shape[0]

    def compute(self, func):
        f = eval(f"self.{func}()")
        return f
    
    #returns min value for each channel
    def min(self):
        z = []
        for c in range(self.n_channels):
            z.append(min(self.trial[c,:]))
        return z
    
    #returns max value for each channel
    def max(self):
        z = []
        for c in range(self.n_channels):
            z.append(max(self.trial[c,:]))
        return z
    
    #returns mean value for each channel
    def mean(self):
        z = []
        for c in range(self.n_channels):
            z.append(np.mean(self.trial[c,:]))
        return z
    
    #returns standard diviation for each channel
    def sd(self):
        z = []
        for c in range(self.n_channels):
            z.append(np.std(self.trial[c,:]))
        return z
    
    #returns the fourier coefficents for a given channel
    def Fourier(self, c):
        coef = fft(self.trial[c, :])
        freq = fftfreq(self.trial.shape[0], 1/512)
        return coef, freq
    
    #returns the three first fourier coefficents for each channel
    def three_first_Fourier(self):
        self.z_1, self.z_2, self.z_3 = [], [], []
        for c in range(self.n_channels):
            coef, freq = self.Fourier(c)
            self.z_1.append(coef[0])
            self.z_2.append(coef[1])
            self.z_3.append(coef[2])          
            # self.z_1.append(int(coef[0]))
            # self.z_2.append(int(coef[1]))
            # self.z_3.append(int(coef[2]))
    
    #First fourier coef
    def first_Fourier(self):
        self.three_first_Fourier()
        return self.z_1
    
    #Second fourier coef
    def second_Fourier(self):
        return self.z_2
    
    #Third fourier coef
    def third_Fourier(self):
        return self.z_3
    
    #returns the frequency of the maimal value of the biggest fourier coefficent
    def max_Fourier(self):
        z = list()
        for c in range(self.n_channels):
            coef, freq = self.Fourier(c)
            index_max = np.argmax(coef)
            z.append(freq[index_max])
        return z
    
    #returns PCA of the whole xi
    def apply_PCA(self):
        n_components = 3
        pca = PCA(n_components)
        X = pca.fit_transform(self.trial)
        return X
    
    def PCA_1(self):
        X = self.apply_PCA()
        return X[0]
    
    def PCA_2(self):
        X = self.apply_PCA()
        return X[1]
    
    def PCA_3(self):
        X = self.apply_PCA()
        return X[2]
    


