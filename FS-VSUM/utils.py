import os
import random
from pathlib import Path
from os import PathLike
from typing import Any, List, Dict
import yaml
import numpy as np
import torch
import json
from pathlib import Path
import numpy as np

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def check_inputs(values,weights,n_items,capacity):
    assert(isinstance(values,list))
    assert(isinstance(weights,list))
    assert(isinstance(n_items,int))
    assert(isinstance(capacity,int))
    assert(all(isinstance(val,int) or isinstance(val,float) for val in values))
    assert(all(isinstance(val,int) for val in weights))
    assert(all(val >= 0 for val in weights))
    assert(n_items > 0)
    assert(capacity > 0)
    
def load(filename):
    return json.load(open(filename, 'r'))

def load_yaml(path: PathLike) -> Any:
    with open(path) as f:
        obj = yaml.safe_load(f)
    return obj

def read_caption(caption_path):
    with open(caption_path,'r') as file:
        caption_lines = [line.strip() for line in file.readlines()]
    return caption_lines

def read_caption_to_list(caption_path):
    with open(caption_path,'r') as file:
        lines = file.readlines()
        lines = [line.strip().rstrip('.') for line in lines if line.strip()]
    return lines

def get_metadata(args,video_name):
    meta_path = f"../datasets/{args.dataset}/metadata.json"
    with open(meta_path,'r') as file:
        metadata = json.load(file)
        
    for item in metadata:
        if (args.dataset == "SumMe") and (item['original title'] == video_name):
            return item['Substituted title']
        elif (args.dataset == "TVSum") and (item['video_id'] == video_name):
            return item['title'], item['genre'], item['query']

#ref https://github.com/HopLee6/SSPVS-PyTorch
def knapsack_dp(values,weights,n_items,capacity,return_all=False):
    check_inputs(values,weights,n_items,capacity)

    table = np.zeros((n_items+1,capacity+1),dtype=np.float32)
    keep = np.zeros((n_items+1,capacity+1),dtype=np.float32)

    for i in range(1,n_items+1):
        assert i <= len(weights), f"Index i out of range: {i}" 
        
        for w in range(0,capacity+1):
            wi = weights[i-1] 
            vi = values[i-1] 
            if (wi <= w) and (vi + table[i-1,w-wi] > table[i-1,w]):
                table[i,w] = vi + table[i-1,w-wi]
                keep[i,w] = 1
            else:
                table[i,w] = table[i-1,w]

    picks = []
    K = capacity

    for i in range(n_items,0,-1):
        if keep[i,K] == 1:
            picks.append(i)
            K -= weights[i-1]

    picks.sort()
    picks = [x-1 for x in picks] 
    if return_all:
        max_val = table[n_items,capacity]
        return picks,max_val
    return picks

#ref https://github.com/boheumd/A2Summ
def get_keyshot_summ(pred: np.ndarray,
                     cps: np.ndarray,
                     n_frames: int,
                     nfps: np.ndarray,
                     picks: np.ndarray,
                     proportion: float = 0.15,
                     seg_score_mode: str = 'mean',
                     method: str = 'knapsack'
                     ) -> np.ndarray:
    """
    Generate keyshot-based video summary i.e. a binary vector.

    :param pred: Predicted importance scores.
    :param cps: Change points, 2D matrix, each row contains a segment.
    :param n_frames: Original number of frames.
    :param nfps: Number of frames per segment.
    :param picks: Positions of subsampled frames in the original video.
    :param proportion: Max length of video summary compared to original length.
    :return: Generated keyshot-based summary.
    """
    picks = np.asarray(picks, dtype=np.int32)
    assert pred.shape == picks.shape, "pred:{} picks:{}".format(pred.shape, picks.shape)
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(len(picks)):
        pos_lo = picks[i]
        pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
        frame_scores[pos_lo:pos_hi] = pred[i]

    # Assign scores to video shots as the average of the frames.
    seg_scores = np.zeros(len(cps), dtype=np.int32)
    for seg_idx, (first, last) in enumerate(cps):
        scores = frame_scores[first:last + 1]
        if seg_score_mode == 'mean':
            seg_scores[seg_idx] = int(1000 * scores.mean())
        elif seg_score_mode == 'sum':
            seg_scores[seg_idx] = int(1000 * scores.sum())

    # Apply knapsack algorithm to find the best shots
    limits = int(round(n_frames * proportion))
    if method == 'knapsack':
        packed = knapsack_dp(seg_scores.tolist(), nfps.tolist(),int(len(cps)), limits)
    elif method == "rank":
        order = np.argsort(seg_scores)[::-1].tolist()
        packed = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                packed.append(i)
                total_len += nfps[i]

    else:
        raise KeyError("Unknown method {}".format(method))

    summary = np.zeros(n_frames, dtype=bool)
    for seg_idx in packed:
        first, last = cps[seg_idx]
        summary[first:last + 1] = True
    return summary, frame_scores

def f1_score(pred: np.ndarray, test: np.ndarray) -> float:
    """Compute F1-score on binary classification task.

    :param pred: Predicted binary label. Sized [N].
    :param test: Ground truth binary label. Sized [N].
    :return: F1-score value.
    """
    assert pred.shape == test.shape
    pred = np.asarray(pred, dtype=bool)
    test = np.asarray(test, dtype=bool)
    overlap = (pred & test).sum()
    if overlap == 0:
        return 0.0
    precision = overlap / pred.sum()
    recall = overlap / test.sum()
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)

def calc_fscore(args,pred,change_points,n_frames,n_frame_per_seg,picks,user_summary,annoindex):
    pred_summary_kp,frame_scores_kp = get_keyshot_summ(pred,change_points,n_frames,n_frame_per_seg,picks,proportion=float(0.15),seg_score_mode=str('mean'),method='knapsack')
    pred_summary_kp = pred_summary_kp.astype(float)

    f1 = f1_score(pred_summary_kp,user_summary[annoindex])
    return f1