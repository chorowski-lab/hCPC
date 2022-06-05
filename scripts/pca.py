#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import glob
import os.path as osp
import numpy as np

import faiss



def get_parser():
    parser = argparse.ArgumentParser(
        description="compute a pca matrix given a folder with pre-computed features."
    )
    # fmt: off
    parser.add_argument('data', help='dir to folder with .npy files containing features')
    parser.add_argument('--output', help='where to save the pca matrix', required=True)
    parser.add_argument('--dim', type=int, help='dim for pca reduction', required=True)
    parser.add_argument('--maxNumFeatures', type=int, help='number of features to use to compute the PCA matrix', default=1000000)
    parser.add_argument('--eigen-power', type=float, default=0, help='eigen power, -0.5 for whitening')
    parser.add_argument('--debug', action='store_true')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(('0.0.0.0', 7310))
        print("Attach debugger now")
        ptvsd.wait_for_attach()

    print("Reading features")
    featurePaths = glob.glob(args.data + "/*.npy")
    np.random.shuffle(featurePaths)
    x = np.array([])
    for featurePath in featurePaths:    
        features = np.load(featurePath, mmap_mode="r")
        x = np.vstack([x, features]) if x.size else features
        if x.shape[0] >= args.maxNumFeatures:
            x = x[:args.maxNumFeatures, :]
            break
    # while x.shape[0] < args.maxNumFeatures:
    #     featurePath = np.random.choice(featurePaths)
    #     features = np.load(featurePath, mmap_mode="r")
    #     features = features[np.random.choice(np.arange(features.shape[0]), size=int(features.shape[0] * 0.1), replace=False), :]
    #     x = np.vstack([x, features]) if x.size else features
    # x = x[:args.maxNumFeatures, :]
    print(x.shape)

    print("Computing PCA")
    pca = faiss.PCAMatrix(x.shape[-1], args.dim, args.eigen_power)
    pca.train(x)
    b = faiss.vector_to_array(pca.b)
    A = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)

    os.makedirs(args.output, exist_ok=True)

    prefix = str(args.dim)
    if args.eigen_power != 0:
        prefix += f"_{args.eigen_power}"
    print(A.T.shape)
    print(b.shape)   
    np.save(osp.join(args.output, f"{prefix}_pca_A"), A.T)
    np.save(osp.join(args.output, f"{prefix}_pca_b"), b)


if __name__ == "__main__":
    main()