# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import numpy as np
import time
import argparse
import sys
import os
import json
from random import shuffle
from cpc.criterion.clustering import kMeanCluster, kMeanGPU
from cpc.dataset import parseSeqLabels
from cpc.model import MultiLevelModel
from pathlib import Path


def getQuantile(sortedData, percent):
    return sortedData[int(percent * len(sortedData))]


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Clustering module using kmeans or dpmeans.')
    parser.add_argument('--pathCheckpoint', type=str,
                        help="Path to the checkpoint of CPC module.")
    parser.add_argument('--pathOutput', type=str,
                        help="Path to the output clustering checkpoint.")
    parser.add_argument('--pathPhone', type=str,
                            help="Path to the aligned phone labels.")
    parser.add_argument(
        '--pathDB', type=str,
        default="/datasets01/LibriSpeech/022219/train-clean-100/")
    parser.add_argument('-k', '--nClusters', type=int, default=50,
                        help="Number of clusters for kmeans algorithm (default: 50).")
    parser.add_argument('-g',  '--nGroups', type=int, default=1,
                        help="Number of groups for kmeans algorithm (default: 1).")
    parser.add_argument('-n', '--MAX_ITER', type=int, default=100,
                        help="Number of iterations (default: 150).")
    parser.add_argument('--recursionLevel', type=int, default=2,
                        help="The speaker recursionLevel in the training dataset (default: 2).")
    parser.add_argument('--extension', type=str, default='.flac',
                        help="The audio file extension (default: .flac).")
    parser.add_argument('--seqList', type=str, default=None,
                        help="Specific the training sequence list (default: None).")
    parser.add_argument('--sizeWindow', type=int, default=10240,
                        help="The size of the window when loading audio data (default: 10240).")
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode, only use a small number of training data.')
    parser.add_argument('--encoder_layer', action='store_true',
                        help='Whether to use the output of the encoder for the clustering.')
    parser.add_argument('--level_gru', type=int, default=None,
                        help='Specify the LSTM hidden level to take the representation (default: None).')
    parser.add_argument('--batchSizeGPU', type=int, default=50,
                        help='Batch size of each GPU (default: 50).')
    parser.add_argument('--DPMean', action='store_true',
                        help='Activate DPMeans training instead of Kmeans.')
    parser.add_argument('-l', '--DPLambda', type=float, default=11,
                        help='Lambda parameter of DPMeans algo (default: 11).')
    parser.add_argument('--perIterSize', type=int, default=-1,
                        help='(Depreciated) Number of items per iteration (default: -1).')
    parser.add_argument('--train_mode', action='store_true',
                        help='Activate training CPC module too.')
    parser.add_argument('--dimReduction', type=str, default=None,
                        help='Dimentionality reduction (default: None)')
    parser.add_argument('--centroidLimits', type=int, nargs=2, default=None,
                        help='centroidLimits when using dimentionality reduction (default: None)')
    parser.add_argument('--getDistanceEstimation', action='store_true',
                        help='Get distance estimation')
    parser.add_argument('--save', action='store_true',
                        help='Save the intermediate checkpoints. The checkpoints will'
                        'be saved in the same directory as the output.')
    parser.add_argument('--load', action='store_true',
                        help='Load the last checkpoint from the same directory as the output.')
    parser.add_argument('--save-last', type=int, default=5,
                        help='Number of last checkpoints to be saved (default: 5).')

    parser.add_argument('--n_process_loader', type=int, default=8,
                          help='Number of processes to call to load the '
                          'dataset')
    parser.add_argument('--max_size_loaded', type=int, default=4000000000,
                          help='Maximal amount of data (in byte) a dataset '
                          'can hold in memory at any given time')

    parser.add_argument('--nullspace', action='store_true',
                          help="Additionally load nullspace")

    parser.add_argument('--pathPCA', type=str,
                        help="Path to the PCA matrices.")

    parser.add_argument('--norm_vec_len', action='store_true',
                        help="Normalize vector lengths.")

    parser.add_argument('--cpcLevel', default=0, type=int,
                        help='Index of the CPC head at to which extract features. ' 
                        'Ignored if get_encoded is True.')

    parser.add_argument('--seqNorm', action='store_true',
                        help='If activated, normalize each batch '
                        'of feature across the time channel before '
                        'computing distances.')

    parser.add_argument('--kmeansppInits', default=0, type=int,
                        help='Number of init rounds using Scitki-learn K-Means++ initialization. If 0 uses dumb standard initialization')

    parser.add_argument('--segmentationMode', type=str, 
                       choices=['cosineDissimilarity', 'collapseRepetitions', 'groundTruth', 
                       'groundTruthWError', 'groundTruthUnder', 'groundTruthOver', 'groundTruthNumSegments', 
                       'boundaryPredictor'],
                       default='cosineDissimilarity')
    parser.add_argument('--segmentOnContext', action='store_true')
    parser.add_argument('--segmentCompression', type=str, default='lstm',
                       choices=['average', 'lstm'],
                       help="Method to use to compress representations within a segment to a single vector.")
    return parser.parse_args(argv)

# some example with nullspace and normalization making dists cosine:
# python cpc/criterion/clustering/clustering_script.py --pathDB /pio/data/zerospeech2021/LibriSpeech/dev-clean \
# --recursionLevel 1 --nClusters 50 --MAX_ITER 10 --level_gru 2 --save --load --batchSizeGPU 200 --max_size_loaded 40000000 \
# --n_process_loader 2 --nullspace --norm_vec_len ../nspChp/64ok/checkpoint_9.pt ../nspChp/tryNew64-11/try11chp.pt


if __name__ == "__main__":
    torch.cuda.empty_cache()

    import os
    from cpc.feature_loader import loadModel, FeatureModule
    from cpc.dataset import findAllSeqs, filterSeqs, AudioBatchData

    args = parseArgs(sys.argv[1:])
    if args.debug:
        import ptvsd
        ptvsd.enable_attach(('0.0.0.0', 7310))
        print("Attach debugger now")
        ptvsd.wait_for_attach()
        args.nGPU = 1
    # Export absolute paths for later use
    args.pathCheckpoint = os.path.abspath(args.pathCheckpoint)    
    args.pathOutput = os.path.abspath(args.pathOutput)    
    args.pathDB = os.path.abspath(args.pathDB)

    if not args.load: 
        assert os.path.exists(args.pathOutput) is False, \
            f"The output file {args.pathOutput} already exists, please check the option --load !"
        assert os.path.exists(os.path.join(os.path.dirname(args.pathOutput), "checkpoint_last.pt")) is False, \
            f"Found last_checkpoint.pt in the output directory, please check the option --load !"

    print(args)
    seqNames, speakers = findAllSeqs([args.pathDB],
                                     speakerLevel=args.recursionLevel,
                                     extension=[args.extension],
                                     loadCache=False)

    phoneLabels = None
    if args.pathPhone is not None:
        print("Loading the phone labels at " + args.pathPhone)
        phoneLabels, nPhones = parseSeqLabels(args.pathPhone)
        print(f"{nPhones} phones found")

    if args.seqList is not None:
        seqNames = filterSeqs(args.seqList, seqNames)
    if args.debug:
        nsamples=1000
        print(f"Debug mode activated, get only {nsamples} samples!")
        shuffle(seqNames)
        seqNames = seqNames[:nsamples]
    if args.getDistanceEstimation:
        shuffle(seqNames)
        seqNames = seqNames[:5000]

    print("")
    print(f'Loading audio data at {args.pathDB}')
    start_time = time.time()
    dataset = AudioBatchData(args.pathDB,
                             args.sizeWindow,
                             seqNames,
                             phoneLabels,
                             len(speakers),
                             nProcessLoader=args.n_process_loader,
                             MAX_SIZE_LOADED=args.max_size_loaded)
    print(f"Dataset loaded in {time.time()-start_time} seconds !")
    print("")

    nGPUs = torch.cuda.device_count()
    batchSize = args.batchSizeGPU * nGPUs
    
    # samplingType = "samespeaker" if args.seqNorm else "uniform" 
    # print(f"Using {samplingType} sampling")
    trainLoader = dataset.getDataLoader(batchSize, "uniform",
                                        False, numWorkers=0)
    print(f"Length of dataLoader: {len(trainLoader)}")
    print("")


    if args.level_gru is None:
        updateConfig = None
    else:
        updateConfig = argparse.Namespace(nLevelsGRU=args.level_gru)

    model, hiddenGAR, hiddenEncoder = loadModel([args.pathCheckpoint], updateConfig=updateConfig, load_nullspace=args.nullspace, pcaPath=args.pathPCA)
    #model = loadModel([args.pathCheckpoint])[0]#, updateConfig=updateConfig)[0]
    print(model)
    
    # Check if dir exists
    if not os.path.exists(os.path.dirname(args.pathOutput)) and os.path.dirname(args.pathOutput):
        Path(os.path.dirname(args.pathOutput)).mkdir(parents=True, exist_ok=True)
    
    segmentationConf = None
    if args.cpcLevel > 0:
        segmentationConf = {
            'segmentationMode': args.segmentationMode,
            'segmentOnContext': args.segmentOnContext,
            'minNumSegments': 1,
            'segmentCompression': args.segmentCompression,
            'featuresDim': hiddenGAR if args.segmentOnContext else hiddenEncoder
        }
    featureMaker = FeatureModule(model, args.encoder_layer, args.cpcLevel, segmentationConf=segmentationConf)
    if args.segmentCompression == 'lstm':
        torch.save(featureMaker.segmenter.state_dict(), os.path.dirname(args.pathOutput) + '/segmenter.pt')
    print("Checkpoint loaded!")
    print("")

    if not args.train_mode:
        featureMaker.eval()
    featureMaker.cuda()    

    pathConfig = f"{os.path.splitext(args.pathOutput)[0]}_args.json"
    with open(pathConfig, 'w') as file:
        json.dump(vars(args), file, indent=2)

    out_state_dict = {}
    print("Starting the clustering...")
    start_time = time.time()
    clusters = kMeanGPU(trainLoader, featureMaker, args.nClusters, args.nGroups,
                            perIterSize=args.perIterSize,
                            MAX_ITER=args.MAX_ITER,
                            save=args.save, load=args.load, 
                            save_dir=os.path.dirname(args.pathOutput),
                            save_last=args.save_last,
                            norm_vec_len=args.norm_vec_len,
                            seqNorm=args.seqNorm,
                            kmeanspp=args.kmeansppInits).cpu()


    print(f'Ran clustering '
          f'in {time.time() - start_time:.2f} seconds')

    clusterModule = kMeanCluster(clusters, norm_vec_len=args.norm_vec_len)

    out_state_dict["state_dict"] = clusterModule.state_dict()
    out_state_dict["encoder_layer"] = args.encoder_layer
    out_state_dict["n_clusters"] = args.nClusters
    out_state_dict['dim'] = clusters.size(2)
    torch.save(out_state_dict, args.pathOutput)
    with open(pathConfig, 'w') as file:
        json.dump(vars(args), file, indent=2)
