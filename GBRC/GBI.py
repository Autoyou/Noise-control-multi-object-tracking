"""
The Gradient Boosting Reconnection Context (GBRC)
mechanism is developed to realize gradient-adaptive
reconnection of the fragment tracks with trajectory drift-
ing noise
"""
import numpy as np
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.ensemble import GradientBoostingRegressor
font = {'family':'Arial','size':20}

def LinearInterpolation(input_, interval):
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]
    output_ = input_.copy()
    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,))
    for row in input_:
        f_curr, id_curr = row[:2].astype(int)
        if id_curr == id_pre:
            if f_pre + 1 < f_curr < f_pre + interval:
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):
                    step = (row - row_pre) / (f_curr - f_pre) * i
                    row_new = row_pre + step
                    output_ = np.append(output_, row_new[np.newaxis, :], axis=0)
        else:
            id_pre = id_curr
        row_pre = row
        f_pre = f_curr
    output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])]
    return output_


def GradientBoostingSmooth(input_, tau):
    output_ = list()
    MAE = []
    MAE1 = []
    ids = set(input_[:, 1])
    for id_ in ids:
        tracks = input_[input_[:, 1] == id_]
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2)
        gpr = GPR(RBF(len_scale, 'fixed'))
        t = tracks[:, 0].reshape(-1, 1)
        x = tracks[:, 2].reshape(-1, 1)
        y = tracks[:, 3].reshape(-1, 1)
        w = tracks[:, 4].reshape(-1, 1)
        h = tracks[:, 5].reshape(-1, 1)
        gpr.fit(t, x)
        xx = gpr.predict(t)[:, 0]
        gpr.fit(t, y)
        yy = gpr.predict(t)[:, 0]
        gpr.fit(t, w)
        ww = gpr.predict(t)[:, 0]
        gpr.fit(t, h)
        hh = gpr.predict(t)[:, 0]
        regr = GradientBoostingRegressor(n_estimators=115,learning_rate=0.065,min_samples_split=6)#learning_rate=0.065,min_samples_split=6,n_estimators=71
        regr.fit(t, x)
        xx = regr.predict(t)
        xx = xx.reshape(-1, 1)
        regr.fit(t, y)
        yy = regr.predict(t)
        yy = yy.reshape(-1, 1)
        regr.fit(t, w)
        ww = regr.predict(t)
        ww = ww.reshape(-1, 1)
        regr.fit(t, h)
        hh = regr.predict(t)
        hh = hh.reshape(-1, 1)
        MAE.append(np.sum((t-gpr.predict(x))**2)/len(t))
        MAE1.append(np.sum((t - regr.predict(x)) ** 2) / len(t))
        output_.extend([
            [t[i, 0], id_, xx[i], yy[i], ww[i], hh[i], 1, -1, -1 , -1] for i in range(len(t))
        ])
    plt.plot(range(1,len(MAE)+1),MAE)
    plt.plot(range(1,len(MAE1)+1),MAE1,color ='green',linestyle='--')
    plt.ylabel('MAE')
    plt.legend(labels=('GSI', 'GBRC'))
    plt.show()
    return output_

# GBI
def GBInterpolation(path_in, path_out, interval, tau):
    input_ = np.loadtxt(path_in, delimiter=',')
    li = LinearInterpolation(input_, interval)
    gbi = GradientBoostingSmooth(li, tau)
    np.savetxt(path_out, gbi, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')