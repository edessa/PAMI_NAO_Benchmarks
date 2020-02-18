import cv2
import torch
import numpy as np
from sklearn.metrics import jaccard_score as jsc
from torch import nn
import csv
import os

def cleanup_obj(image_files, objs):
    interested_uids = []
    filename_list = []
    selected_filepath = []
    object_list  = []
    person_list = []
    selected_uids = []
    for i in range(len(image_files)):
        uidname = image_files[i].split('_')[0].split('/')[3]
        filename = image_files[i]
        interested_uids.append(uidname)
        filename_list.append(filename)
    my_list = list(set(interested_uids))

    with open('./evaluations/EPIC_train_action_labels.csv') as f:
        reader = list(csv.DictReader(f))
        for m in my_list:
            for row in reader:
                if int(m)==int(row['uid']):
                    person=row['participant_id']
                    actions = row['verb_class']
                    objectclass=row['noun_class']
                    object_list.append(objectclass)
                    person_list.append(person)
    unique_object = np.unique(np.array(object_list))
    a = {}

    with open('./evaluations/EPIC_train_action_labels.csv') as f:
        reader = list(csv.DictReader(f))
        for row in reader:
            if int(row['noun_class']) in objs:
                if row['uid'] in my_list:
                    if int(row['noun_class']) not in a.keys():
                        a[int(row['noun_class'])] = 1
                    else:
                        a[int(row['noun_class'])] += 1
                    selected_uids.append(int(row['uid']))

    unique_selected_uids = np.unique(np.array(selected_uids))
    for i in range(len(image_files)):
        uidname = image_files[i].split('_')[0].split('/')[3]
        filename=image_files[i]
        if int(uidname) in unique_selected_uids:
            selected_filepath.append(i)

    return selected_filepath, a

def loss_seg_fn(output, target):
    weight = torch.tensor([1.0]).cuda()
    loss_fn = nn.BCELoss()
    loss = loss_fn(output.cuda(), target.cuda())
    return loss

def l1(output, target):
    res = (target != -1).nonzero()
    loss = torch.mean(torch.abs(output[res] - target[res]))
    return loss

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
