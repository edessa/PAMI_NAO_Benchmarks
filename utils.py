import cv2
import numpy as np
from sklearn.metrics import jaccard_score as jsc

def KL(P,Q):
    epsilon = 0.00001
    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon
    divergence = np.sum(P*np.log(P/Q))
    return divergence

def coherence_error(flow, pred_1, pred_2):
    pred_1 = pred_1.reshape(128, 228)
    projected = cv2.remap(pred_1, flow, None, cv2.INTER_NEAREST)
    return jsc(pred_2, projected.reshape(-1,))

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def histogram_intersection(h1, h2):
    n1 = normalize(h1)
    n2 = normalize(h2)
    sm = 0
    for i in range(len(h1)):
        sm += min(n1[i], n2[i])
    return sm
