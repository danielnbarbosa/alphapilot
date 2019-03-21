'''
Functions to help in calculating mAP and generating results files.
'''

import fastai
import pandas as pd
import json
import torch
from pathlib import Path
from scorer import *

mAP_scorer = mAPScorer()


# calculate inference time
def inference_time(preds, tic, toc):
    total = (toc-tic)
    avg = total / len(preds)
    print('total inference time: {:.2f} seconds'.format(total))
    print('average inference time: {:.2f} seconds'.format(avg))


# convert (y, x) coodinates from a canvas in range ([-1, 1], [-1, 1])
#  to (x, y) coordinates on a canvas in range ([0, 1296], [0, 864])
def convert_coords(input_coords):
    # copy input to new tensor
    coords = input_coords.clone().detach()
    # scale based on real image size (1296x864)
    coords = coords.reshape(4,2)
    coords[:,1] = ((coords[:,1] + 1) * (1296 / 2 ))  # x
    coords[:,0] = ((coords[:,0] + 1) * (864 / 2 ))  # y
    # swap x (second column) and y (first column)
    coords = torch.stack([coords[:,1], coords[:,0]], dim=1)
    # flatten and return as list
    return list(coords.flatten().numpy())

# convert dataframe to JSON format used by AlphaPilot and write file
def write_json(df: pd.DataFrame, path: Path):
    # convert dataframe to JSON string and strip out outer '[]'
    json_str = df.to_json(orient='records')[1:-1]
    # write to file
    f = open(path, 'w')
    f.write(json_str)
    f.close()


# write predictions JSON file
def write_preds_json(ds: fastai.data_block.LabelList, preds: torch.Tensor, size: tuple, path: Path):
    result = {}
    df = ds.to_df()['x']
    for i in range(len(preds)):
        fname = df[i].name
        coords = convert_coords(preds[i])
        confidence = 1.0  # only one box per image
        result[fname] = [[coords + [confidence]]]
    # load into dataframe and write to file
    write_json(pd.DataFrame.from_dict(result), path)


# write labels JSON file
def write_labels_json(preds_path: Path, labels_path: Path, path: Path):
    df_preds = pd.read_json(preds_path)
    df_labels_all = pd.read_json(labels_path)
    df_labels = pd.DataFrame()
    for fname in df_preds.keys():
        df_labels[fname] = [df_labels_all[fname]]
    # write to file
    write_json(df_labels, path)


# given labels JSON and prediction JSON generate mAP score
# taken from code provided by AlphaPilot
def score(labels_fname: Path, preds_fname: Path):
    with open(labels_fname,'r') as f:
        GT_data = json.load(f)
    with open(preds_fname,'r') as f:
        pred_data = json.load(f)

    n_GT = mAP_scorer.countBoxes(GT_data)
    n_Pred = mAP_scorer.countBoxes(pred_data)

    coco_score = mAP_scorer.COCO_mAP(GT_data,pred_data)

    print("COCO mAP for detector is {}".format(coco_score))
