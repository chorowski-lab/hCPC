# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import sys
import torch
import json
import time
import numpy as np
from pathlib import Path
from copy import deepcopy
import os
import tqdm
import random

import cpc.criterion as cr
import cpc.criterion.soft_align as sa
import cpc.feature_loader as fl
import cpc.utils.misc as utils
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels
from cpc.model import CPCModelNullspace, MultiLevelModel, Segmenter
# import pandas as pd

def deltas(D):
    res = [0]
    for i in range(1, D+1):
        if random.random() < 0.5:
            res.append(i)
            res.append(-i)
        else:
            res.append(-i)
            res.append(i)
    return res
    

def run(featureMaker,
        dataLoader,
        pathCheckpoint,
        onEncodings,
        toleranceInFrames=2,
        strict=False):
    print("%d batches" % len(dataLoader))

    featureMaker.eval()
    logs = {"precision": 0, "recall": 0, "f1": 0, "r": 0}
    EPS = 1e-7
    if  isinstance(featureMaker, MultiLevelModel):
        segmenter = featureMaker.segmenter
        model = featureMaker.frameLevelModel
    else:
        model = featureMaker
        segmenter = Segmenter(
            'cosineDissimilarity',
            not onEncodings,
            1,
            'average')
    # results = []
    for step, fulldata in tqdm.tqdm(enumerate(dataLoader)):
        with torch.no_grad():
            batchData, labelData = fulldata
            label = labelData['phone']
            cFeature, encodedData, label, extraLosses = model(batchData.cuda(), label.cuda())
            seqEndIdx = torch.arange(0, encodedData.size(0)*encodedData.size(1) + 1, encodedData.size(1)).cuda()
            
            predictedBoundaries = segmenter(cFeature, encodedData, label.cuda(), returnBoundaries=True).bool()
            predictedBoundaries = torch.nonzero(predictedBoundaries.view(-1), as_tuple=True)[0]
            # Ensure that minibatch boundaries are preserved
            predictedBoundaries = torch.unique(torch.cat((predictedBoundaries, seqEndIdx)), sorted=True)

        maxRval = -np.inf
        diffs = torch.diff(label, dim=1)
        phone_changes = torch.cat((torch.ones((label.shape[0], 1), device=label.device), diffs), dim=1)
        trueBoundaries = torch.nonzero(phone_changes.contiguous().view(-1), as_tuple=True)[0]
        # Ensure that minibatch boundaries are preserved
        trueBoundaries = torch.unique(torch.cat((trueBoundaries, seqEndIdx)), sorted=True)

        if strict:
            true_positive = 0
            f_cnt = g_cnt = 0

            fb = predictedBoundaries.tolist()
            gb = trueBoundaries.tolist()
            gb = set(gb)
            random.shuffle(fb)

            g_cnt += len(gb)
            f_cnt += len(fb)


            for n in fb:
                for delta in deltas(toleranceInFrames):
                    if n + delta in gb:
                        gb -= {n+delta}
                        true_positive += 1
                        break
            
            precision = torch.Tensor([true_positive / (f_cnt + EPS)])
            recall = torch.Tensor([true_positive / (g_cnt + EPS)])
            f1 = 2 * (precision * recall) / (precision + recall + EPS)
            os = recall / (precision + EPS) - 1
            r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
            r2 = (-os + recall - 1) / (np.sqrt(2))
            rVal = 1 - (np.abs(r1) + np.abs(r2)) / 2
        else:
            precisionCounter = 0
            recallCounter = 0

            for predictedBoundary in predictedBoundaries:
                minDist = torch.min(torch.abs(trueBoundaries - predictedBoundary))
                precisionCounter += (minDist <= toleranceInFrames)

            for trueBoundary in trueBoundaries:
                minDist = torch.min(torch.abs(predictedBoundaries - trueBoundary))
                recallCounter += (minDist <= toleranceInFrames)

            precision = precisionCounter / (len(predictedBoundaries) + EPS)
            recall = recallCounter / (len(trueBoundaries) + EPS)
            f1 = 2 * (precision * recall) / (precision + recall + EPS)
            os = recall / (precision + EPS) - 1
            r1 = torch.sqrt((1 - recall) ** 2 + os ** 2)
            r2 = (-os + recall - 1) / (np.sqrt(2))
            rVal = 1 - (torch.abs(r1) + torch.abs(r2)) / 2

        logs["precision"] += precision.view(1).cpu().numpy()
        logs["recall"] += recall.view(1).cpu().numpy()
        logs["f1"] += f1.view(1).cpu().numpy()
        logs["r"] += rVal.view(1).cpu().numpy()
    logs = utils.update_logs(logs, step)

    utils.show_logs("Results", logs)
    for key, value in dict(logs).items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        logs[key] = value
    utils.save_logs(logs, f"{pathCheckpoint}_logs.json")



def parse_args(argv):
    parser = argparse.ArgumentParser(description='Phoneme segmentation test')
    parser.add_argument('--pathDB', type=str, nargs="+",
                        help="Path to the directory containing the audio data.")
    parser.add_argument('--pathVal', type=str, nargs="+", default=None,
                          help='Path(s) to a .txt file containing the list of the '
                          'validation sequences.')
    parser.add_argument('--load', type=str, nargs='*',
                        help="Path to the checkpoint to evaluate.")
    parser.add_argument('--pathPhone', type=str, default=None,
                        help="Path to the phone labels.")
    # parser.add_argument('--pathPredictions', type=str, default=None,
    #                     help="Path to the predicted labels. If given, will"
    #                     " ignore the checkpoints.")
    # parser.add_argument('--pathWords', type=str, default=None,
    #                     help="Path to the word labels. If given, will"
    #                     " compute word separability.")
    parser.add_argument('--pathCheckpoint', type=str, default='out',
                        help="Path of the output directory where the "
                        " checkpoints should be dumped.")
    # parser.add_argument('--nGPU', type=int, default=-1,
    #                     help='Bumber of GPU. Default=-1, use all available '
    #                     'GPUs')
    parser.add_argument('--batchSizeGPU', type=int, default=8,
                        help='Batch size per GPU.')
    parser.add_argument('--debug', action='store_true',
                        help='If activated, will load only a small number '
                        'of audio data.')
    parser.add_argument('--file_extension', type=str, nargs="+", default=".flac",
                        help="Extension of the audio files in pathDB.")
    parser.add_argument('--get_encoded', action='store_true',
                        help="If activated, will work with the output of the "
                        " convolutional encoder (see CPC's architecture).")
    parser.add_argument('--ignore_cache', action='store_true',
                        help="Activate if the sequences in pathDB have"
                        " changed.")
    parser.add_argument('--size_window', type=int, default=20480,
                        help="Number of frames to consider in each batch.")
    parser.add_argument('--nProcessLoader', type=int, default=8,
                          help='Number of processes to call to load the '
                          'dataset')
    parser.add_argument('--max_size_loaded', type=int, default=4000000000,
                          help='Maximal amount of data (in byte) a dataset '
                          'can hold in memory at any given time')
    parser.add_argument('--tolerance', type=int, default=2,
                          help='Tolerance in frames to count as a boundary match')
    parser.add_argument('--strict', action='store_true',
                        help="Strict one to one boundary matching.")
    # parser.add_argument("--model", type=str, default="cpc",
    #                       help="Pre-trained model architecture ('cpc' [default] or 'wav2vec2').")
    parser.add_argument('--segmentationMode', type=str, 
                        choices=['cosineDissimilarity', 'collapseRepetitions', 'groundTruth', 
                        'groundTruthWError', 'groundTruthUnder', 'groundTruthOver', 'groundTruthNumSegments', 
                        'groundTruthUnderMixed', 'groundTruthOverMixed', 'boundaryPredictor'],
                        default='cosineDissimilarity')

    args = parser.parse_args(argv)
    # if args.nGPU < 0:
    #     args.nGPU = torch.cuda.device_count()

    args.load = [str(Path(x).resolve()) for x in args.load]
    # args.pathCheckpoint = str(Path(args.pathCheckpoint).resolve())

    return args


def main(argv):
    args = parse_args(argv)

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(('0.0.0.0', 7310))
        print("Attach debugger now")
        ptvsd.wait_for_attach()

    phoneLabels, nPhones = parseSeqLabels(args.pathPhone)
    # wordLabels = None
    # if args.pathWords is not None:
    #     wordLabels, nWords = parseSeqLabels(args.pathWords)
    
    seqNames, speakers = findAllSeqs(args.pathDB,
                                    extension=args.file_extension,
                                    loadCache=False)
    if args.pathVal is not None:
        seqNames = filterSeqs(args.pathVal, seqNames)

    model, hiddenGAR, hiddenEncoder = fl.loadModel(args.load)

    model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=range(args.nGPU))

    # Dataset
    if args.debug:
        seqNames = seqNames[:100]

    db = AudioBatchData(args.pathDB, args.size_window, seqNames,
                        phoneLabels, len(speakers), nProcessLoader=args.nProcessLoader)

    batchSize = args.batchSizeGPU # * args.nGPU
    dataLoader = db.getDataLoader(batchSize, 'sequential', False, numWorkers=0)

    # Checkpoint directory
    pathCheckpoint = Path(args.pathCheckpoint)
    pathCheckpoint.mkdir(exist_ok=True)
    pathCheckpoint = str(pathCheckpoint / "checkpoint")

    # with open(f"{pathCheckpoint}_args.json", 'w') as file:
    #     json.dump(vars(args), file, indent=2)

    run(model, dataLoader, pathCheckpoint, args.get_encoded, args.tolerance, args.strict)



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
