"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet

import sys
sys.path.append("/jet/prs/workspace/rxrx1-utils")
import rxrx.io as rio

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far

def get_image_path(basepath_data, original_image_size, dataset, id_code, site):
    return basepath_data + f"rgb_{original_image_size}/" + f"{dataset}/" + f"{id_code}_s{site}.jpg"

def get_paths_control(original_image_size=512, basepath_data="/jet/prs/workspace/data/",
                      n_pos_pairs=90, n_neg_pairs=90):
    
    # load image metadata
    print("Reading RXRX1 metadata...")
    df = rio.combine_metadata().reset_index()

    # get control samples only
    df_contr = df[(df["dataset"]=="test") & (df["well_type"]=="positive_control")]
    print(f"Metadata dataframe for TEST POSITIVE CONTROL wells shape: {df_contr.shape}")

    df_contr["image_path"] = df.apply(lambda row: get_image_path(basepath_data, 
                                      original_image_size, "test", 
                                      row["id_code"], row["site"]), axis=1)
    path_list = []
    issame_list = []
    sirnas = df_contr["sirna"].unique()
    for i in range(n_pos_pairs):
        sirna = np.random.choice(sirnas, size=1)[0]
        (p0, p1) = np.random.choice(df_contr.loc[df_contr["sirna"]==sirna, "image_path"].values, size=2, replace=False)
        path_list += (p0, p1)
        issame_list.append(True)
    for i in range(n_neg_pairs):
        (sirna0, sirna1) = np.random.choice(sirnas, size=2, replace=False)
        p0 = np.random.choice(df_contr.loc[df_contr["sirna"]==sirna0, "image_path"].values, size=1)[0]
        p1 = np.random.choice(df_contr.loc[df_contr["sirna"]==sirna1, "image_path"].values, size=1)[0]
        path_list += (p0, p1)
        issame_list.append(False)
    
    return path_list, issame_list
  
def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)



