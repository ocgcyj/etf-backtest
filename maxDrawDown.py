# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 00:22:21 2018

@author: kaychen
"""
import numpy as np
import matplotlib.pyplot as plt 


def maxDrawDownChg(L):
    max_val = 0
    max_drawdown =0
    for i in range(0, len(L)):
        if L[i] > max_val:
            max_val = L[i]
        
        if max_val - L[i] > max_drawdown:
            max_drawdown = max_val - L[i]
            if i > 0:
                max_drawdown_chg = max_drawdown / max_val
    return max_drawdown_chg
    


n = 1000
xs = np.random.randn(n).cumsum()
i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
j = np.argmax(xs[:i]) # start of period
maxdropdownchg = (xs[j] - xs[i]) / xs[j]

plt.plot(xs)
plt.plot([i, j], [xs[i], xs[j]], 'o', color='Red', markersize=10)