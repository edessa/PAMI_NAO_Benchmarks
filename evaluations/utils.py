import cv2
import torch
import numpy as np
from sklearn.metrics import jaccard_score as jsc
from scipy import spatial
from torch import nn
import csv
import os

def cleanup_obj(image_files, objs, filename='./evaluations/EPIC_train_action_labels.csv'):
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
    noun_hist = {}
    uid_to_obj = {}
    with open('./evaluations/EPIC_train_action_labels.csv') as f:
        reader = list(csv.DictReader(f))
        for row in reader:
            uid_to_obj[row['uid']] = row['noun_class']
            if int(row['noun_class']) in objs:
                if row['uid'] in my_list:
                    if int(row['noun_class']) not in noun_hist.keys():
                        noun_hist[int(row['noun_class'])] = 1
                    else:
                        noun_hist[int(row['noun_class'])] += 1
                    selected_uids.append(int(row['uid']))

    unique_selected_uids = np.unique(np.array(selected_uids))
    for i in range(len(image_files)):
        uidname = image_files[i].split('_')[0].split('/')[3]
        filename=image_files[i]
        if int(uidname) in unique_selected_uids:
            selected_filepath.append(i)

    return selected_filepath, noun_hist, uid_to_obj

def get_judd_auc(output_mask, gt_mask):
    gt_mask_jacc = np.array(gt_mask) > 0.5
    output_rel_gt = output_mask[gt_mask_jacc]
    n_f = len(output_rel_gt)
    n_p = len(output_mask)

    thresholds = sorted(output_rel_gt)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)

    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1

    for k, thresh in enumerate(thresholds):
        above_th = np.sum(output_mask >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_f) # Ratio saliency map values at fixation locations above threshold
        fp[k+1] = (above_th - k - 1) / float(n_p - n_f) # Ratio other saliency map values above threshold
    return np.trapz(tp, fp) # y, x

def loss_seg_fn(output, target):
    weight = torch.tensor([1.0]).cuda()
    loss_fn = nn.BCELoss()
    loss = loss_fn(output.cuda(), target.cuda())
    return loss

def l1(output, target):
    res = (target != -1).nonzero()
    loss = torch.mean(torch.abs(output[res] - target[res]))
    return loss

def KLD(map1, map2, eps = 1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    kld = np.sum(map2*np.log( map2/(map1+eps) + eps))
    return kld

def coherence_error(flow, pred_1, pred_2):
    pred_1 = pred_1.reshape(128, 228)
    projected = cv2.remap(pred_1, flow, None, cv2.INTER_NEAREST)
    return jsc(pred_2, projected.reshape(-1,))

def normalize(v):
    norm = sum(v)
    if norm == 0:
       return v
    return v / float(norm)

def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)
