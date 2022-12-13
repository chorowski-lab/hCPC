# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse


def get_default_cpc_config():
    parser = set_default_cpc_config(argparse.ArgumentParser())
    return parser.parse_args([])


def set_default_cpc_config(parser):
    # Run parameters

    group = parser.add_argument_group('Architecture configuration',
                                      description="The arguments defining the "
                                      "model's architecture.")
    group.add_argument('--hiddenEncoder', type=int, default=256,
                       help='Hidden dimension of the encoder network.')
    group.add_argument('--hiddenEncoderSegment', type=int, default=256,
                       help='Hidden dimension of the encoder network of the segment-level model.')
    group.add_argument('--linearOutput', action='store_true')
    group.add_argument('--hiddenGar', type=int, default=256,
                       help='Hidden dimension of the auto-regressive network')
    group.add_argument('--hiddenGarSegment', type=int, default=256,
                       help='Hidden dimension of the auto-regressive network of the segment-level model.')
    group.add_argument('--nPredicts', type=int, default=12,
                       help='Number of steps to predict.')
    group.add_argument('--nPredictsSegment', type=int, default=1,
                       help='Number of steps to predict on second head.')

    group.add_argument('--segmentLevel', action='store_true', help="Model at segment level.")
    group.add_argument('--multiLevel', action='store_true', help="Model at frame and segment level.")
    group.add_argument('--segmentCompression', type=str, default='average',
                       choices=['average', 'lstm', 'random'],
                       help="Method to use to compress representations within a segment to a single vector.")
    group.add_argument('--normalizeCPCScore', action='store_true',
                       help="Uses cosine similarity in the contrastive loss.")
    group.add_argument('--targetQuantizer', type=str, default='none',
                       choices=['gumbel', 'kmeans', 'robustKmeans', 'none'],
                       help="Architecture to use for quantizer of the targets on the CPC loss.")
    
    group.add_argument('--encodingsQuantizer', type=str, default='none',
                       choices=['gumbel', 'kmeans', 'robustKmeans', 'none'],
                       help="Architecture to use for quantizer of the outputs of the convolutional encoder.")
    group.add_argument('--contextQuantizer', type=str, default='none',
                       choices=['gumbel', 'kmeans', 'robustKmeans', 'none'],
                       help="Architecture to use for quantizer of the context vectors.")
    group.add_argument('--targetQuantizerSegment', type=str, default='none',
                       choices=['gumbel', 'kmeans', 'robustKmeans', 'none'],
                       help="Architecture to use for quantizer of the targets on the CPC loss at the segment level.")
    group.add_argument('--numGroupsCodebook', type=int, default=1)
    group.add_argument('--numCodesCodebook', type=int, default=512)
    group.add_argument('--adjacentNegatives', action='store_true',
                       help="If true takes the negative as the immediate adjacent next samples.")

    group.add_argument('--CPCCTC', action='store_true')
    group.add_argument('--CPCCTCNumMatched', type=int, default=16)
    group.add_argument('--CPCCTCNumMatchedSegment', type=int, default=1)
    group.add_argument('--CPCCTCSkipBeg', type=int, default=0)
    group.add_argument('--CPCCTCSkipEnd', type=int, default=0)
    group.add_argument('--CPCCTCSelfLoop', action='store_true')
    group.add_argument('--CPCCTCLearnBlank', action='store_true')
    group.add_argument('--CPCCTCNoNegsMatchWin', action='store_true')
    group.add_argument('--CPCCTCMasq', default="")
    group.add_argument('--CPCCTCLossTemp', type=float, default=1.0)
    group.add_argument('--CPCCTCNormalizeEncs', action='store_true')
    group.add_argument('--CPCCTCNormalizePreds', action='store_true')
    group.add_argument('--limitNegsInBatch', type=int, default=0,
                       help='Limit the number of different seqs from whithc neg samples are taken.')

    group.add_argument('--headWeights', type=float, nargs="+", default=[1.0, 1.0])
    group.add_argument('--segmentationMode', type=str, 
                       choices=['cosineDissimilarity', 'collapseRepetitions', 'groundTruth', 
                       'groundTruthWError', 'groundTruthUnder', 'groundTruthOver', 'groundTruthNumSegments', 
                       'groundTruthUnderMixed', 'groundTruthOverMixed', 'boundaryPredictor'],
                       default='cosineDissimilarity')
    group.add_argument('--rlSetup', type=str, 
                       choices=['vanillaReinforce', 'reinforceWBaseline', 'reinforceWAdvantage'],
                       default='reinforceWBaseline')
    group.add_argument('--segmentOnContext', action='store_true')
    group.add_argument('--freezeFrameModel', action='store_true')
    group.add_argument('--loadOnlyFrameModel', action='store_true')
    
    group.add_argument('--negativeSamplingExt', type=int, default=128,
                       help='Number of negative samples to take.')
    group.add_argument('--negativeSamplingExtSegment', type=int, default=1,
                       help='Number of negative samples to take on segment-level head.')
    group.add_argument('--learningRate', type=float, default=2e-4)
    group.add_argument('--codebookLearningRate', type=float, default=2e-3)
    group.add_argument('--schedulerStep', type=int, default=-1,
                       help='Step of the learning rate scheduler: at each '
                       'step the learning rate is divided by 2. Default: '
                       'no scheduler.')
    group.add_argument('--schedulerRamp', type=int, default=None,
                       help='Enable a warm up phase for the learning rate: '
                       'adds a linear ramp of the given size.')
    group.add_argument('--beta1', type=float, default=0.9,
                       help='Value of beta1 for the Adam optimizer')
    group.add_argument('--beta2', type=float, default=0.999,
                       help='Value of beta2 for the Adam optimizer')
    group.add_argument('--epsilon', type=float, default=1e-08,
                       help='Value of epsilon for the Adam optimizer')
    group.add_argument('--sizeWindow', type=int, default=20480,
                       help='Number of frames to consider at each batch.')
    group.add_argument('--nEpoch', type=int, default=200,
                       help='Number of epoch to run')
    group.add_argument('--samplingType', type=str, default='samespeaker',
                       choices=['samespeaker', 'uniform',
                                'samesequence', 'sequential'],
                       help='How to sample the negative examples in the '
                       'CPC loss.')
    group.add_argument('--nLevelsPhone', type=int, default=1,
                       help='(Supervised mode only). Number of layers in '
                       'the phone classification network.')
    group.add_argument('--cpc_mode', type=str, default=None,
                       choices=['reverse', 'none'],
                       help='Some variations on CPC.')
    group.add_argument('--encoder_type', type=str,
                       choices=['cpc', 'mfcc', 'lfb'],
                       default='cpc',
                       help='Replace the encoder network by mfcc features '
                       'or learned filter banks')
    group.add_argument('--normMode', type=str, default='layerNorm',
                       choices=['instanceNorm', 'ID', 'layerNorm',
                                'batchNorm'],
                       help="Type of normalization to use in the encoder "
                       "network (default is layerNorm).")
    group.add_argument('--onEncoder', action='store_true',
                       help="(Supervised mode only) Perform the "
                       "classification on the encoder's output.")
    group.add_argument('--random_seed', type=int, default=None,
                       help="Set a specific random seed.")
    group.add_argument('--speakerEmbedding', type=int, default=0,
                       help="(Depreciated) Feed the prediction network with "
                       "speaker embeddings along with the usual sequence.")
    group.add_argument('--arMode', default='LSTM',
                       choices=['GRU', 'LSTM', 'RNN', 'no_ar', 'transformer'],
                       help="Architecture to use for the auto-regressive "
                       "network (default is lstm).")
    group.add_argument('--NoARonRegHead', action='store_true')
    group.add_argument('--nLevelsGRU', type=int, default=1,
                       help='Number of layers in the autoregressive network.')
    group.add_argument('--rnnMode', type=str, default='transformer',
                       choices=['transformer', 'RNN', 'LSTM', 'linear',
                                'ffd', 'conv4', 'conv8', 'conv12', 'none'],
                       help="Architecture to use for the prediction network")
    group.add_argument('--rnnModeSegment', type=str, default='transformer',
                       choices=['transformer', 'RNN', 'LSTM', 'linear',
                                'ffd', 'conv4', 'conv8', 'conv12', 'none'],
                       help="Architecture to use for the prediction network at segment level")
    group.add_argument('--dropout', action='store_true',
                       help="Add a dropout layer at the output of the "
                       "prediction network.")
    group.add_argument('--abspos', action='store_true',
                       help='If the prediction network is a transformer, '
                       'active to use absolute coordinates.')
    group.add_argument('--sincNet', action='store_true',
                       help='Use a sincNet layer as first layer of the convolutional encoder.')
    group.add_argument('--nLayersBoundaryPredictor', type=int, default=1,
                       help='Number of layers in the boundary predictor (if used).')
    group.add_argument('--noSegmentation', action='store_true',
                       help='Not to use segmentation in a multi-level model.')
    return parser
