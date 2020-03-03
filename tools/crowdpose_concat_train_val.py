# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Concat CrowdPose train and val')

    parser.add_argument('--data_dir',
                        help='data directory containing json annotation file',
                        default='data/crowd_pose/json',
                        type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    train_dataset = json.load(open(os.path.join(args.data_dir, 'crowdpose_train.json')))
    val_dataset = json.load(open(os.path.join(args.data_dir, 'crowdpose_val.json')))

    trainval_dataset = {}
    trainval_dataset['categories'] = train_dataset['categories']
    trainval_dataset['images'] = []
    trainval_dataset['images'].extend(train_dataset['images'])
    trainval_dataset['images'].extend(val_dataset['images'])
    trainval_dataset['annotations'] = []
    trainval_dataset['annotations'].extend(train_dataset['annotations'])
    trainval_dataset['annotations'].extend(val_dataset['annotations'])

    with open(os.path.join(args.data_dir, 'crowdpose_trainval.json'), 'w') as f:
        json.dump(trainval_dataset, f)


if __name__ == '__main__':
    main()
