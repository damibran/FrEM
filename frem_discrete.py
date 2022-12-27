import numpy as np
import torchxrayvision as xrv
import pandas as pd
from multiprocessing import Pool
from functools import partial
import os
import datetime

class image_mat_sampler:
    def __init__(self, img_mat: np.ndarray):
        self.img_mat = img_mat
        self.N = max(img_mat.shape)  # min - edge cut (inscribed circle), max - edge extend (described circle)

        if self.N % 2 != 0: # idk how to handle centre element, so just make size even
            self.N += 1

    def getElmnt(self, q, p):
        if (q < self.img_mat.shape[0] and p < self.img_mat.shape[1]) and (self.x(q)**2 + self.y(p)**2 <= 1):
            return self.img_mat[q][p]
        else:
            return 0

    def x(self,q):
        return (2*q - self.N + 1)/self.N

    def y(self,p):
        return (self.N - 1 - 2*p)/self.N

def frem(img: image_mat_sampler, t, n, m):

    def Tp(r):
        return (r**(t-1)) * np.sqrt(2/(r**t)) * np.exp(1j*2*n*np.pi*(r**t))

    def x(q):
        return (2*q - img.N + 1)/img.N

    def y(p):
        return (img.N - 1 - 2*p)/img.N

    def r(q, p):
        return np.sqrt(x(q)**2 + y(p)**2)

    def theta(q, p):
        y_v = y(p)
        x_v = x(q)

        angle = np.arctan2(y_v,x_v)

        angle = (2*np.pi + angle) * (angle < 0) + angle*(angle >= 0) # transorm from [-pi;pi] to [0;2pi]

        return angle

    def under_2_sum(q,p):
        return img.getElmnt(q, p) * np.conjugate(Tp(r(q, p))) * np.exp(-1j*m*theta(q, p))

    def under_1_sum(p):
        return sum(under_2_sum(q, p) for q in range(img.N))

    sum_v = sum(under_1_sum(p) for p in range(img.N))

    return sum_v/(np.pi*img.N**2)

# usage example
if __name__ == '__main__':

    d = xrv.datasets.COVID19_Dataset(
        imgpath="covid-chestxray-dataset/images/", csvpath="covid-chestxray-dataset/metadata.csv")

    outputFile = 'new_data.csv'

    Nmax = 9
    alphas = [1.5]
    data = dict()
    for a in alphas:
        for n in range(Nmax):
            for m in range(Nmax):
                data["FrEM_"+str(a)+"_"+str(n)+"_"+str(m)
                     + "_Re"] = 0
                data["FrEM_"+str(a)+"_"+str(n)+"_"+str(m)
                     + "_Im"] = 0

    data['COVID'] = 0

    remained_d =[]
    for i in range(187,len(d)):
        remained_d.append(d[i])

    remained = list(zip(range(187,len(d)),remained_d))

    remained.sort(key= lambda v: max(v[1]["img"][0].shape))

    pool = Pool()
    for i in range(len(d)):

        sample = d[i]
        sample_img = image_mat_sampler(sample["img"][0])

        print('Time:',datetime.datetime.now(),'Img_N:',i)

        for a in alphas:
            for n in range(Nmax):
                res = pool.map(
                    partial(frem, sample_img, a, n), range(Nmax))
                for m in range(Nmax):
                    val = res[m]
                    data["FrEM_"+str(a)+"_"+str(n)+"_"+str(m)
                         + "_Re"] = np.real(val)
                    data["FrEM_"+str(a)+"_"+str(n)+"_"+str(m)
                         + "_Im"] = np.imag(val)

        data["COVID"] = sample["lab"][3]

        pd.DataFrame(data, index=[i]).to_csv(outputFile, mode='a',
                                             index=True, header=not os.path.exists(outputFile))
