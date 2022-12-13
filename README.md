This code is based on `CPC_audio` (<https://github.com/facebookresearch/CPC_audio>) and it implements the Contrast Predictive Coding algorithm on audio data, as described in the paper [Unsupervised Pretraining Transfers well Across Languages](https://arxiv.org/abs/2002.02848). This is an unsupervised method to train audio features directly from the raw waveform.

## Setup instructions

1/ Install libraries which would be required for torch-audio https://github.com/pytorch/audio :
 * MacOS: `brew install sox`
 * Linux: `sudo apt-get install sox libsox-dev libsox-fmt-all`

2/ `conda env create -f environment.yml && conda activate cpc37`

3/ Run setup.py
`python setup.py develop`

### Standard datasets

We suggest to train the model either on [Librispeech](http://www.openslr.org/12/) or [libri-light](https://github.com/facebookresearch/libri-light).


## How to run a session

To run a training session:

* CPC:

```bash 
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION
--pathCheckpoint $PATH_CHECKPOINT_DIR --normMode layerNorm --dropout --n_process_loader 1 --batchSizeGPU 32 --nPredicts 12 --limitNegsInBatch 8 --nEpoch 50 --nGPU 2 --nLevelsGRU 2 --schedulerRamp 10 --normalizeCPCScore
```

* ACPC:

```bash 
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION
--pathCheckpoint $PATH_CHECKPOINT_DIR --normMode layerNorm --dropout --n_process_loader 1 --batchSizeGPU 32 --CPCCTC --nPredicts 6 --CPCCTCNumMatched 12 --limitNegsInBatch 8 --nEpoch 50 --nGPU 2 --nLevelsGRU 2 --schedulerRamp 10 --normalizeCPCScore
```

* SCPC:

```bash 
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION
--pathCheckpoint $PATH_CHECKPOINT_DIR --normMode layerNorm --dropout --n_process_loader 1 --batchSizeGPU 32 --nPredicts 1 --limitNegsInBatch 8 --nEpoch 50 --nGPU 2 --schedulerRamp 10 --rnnMode none --arMode no_ar --negativeSamplingExt 1 --nPredicts 1 --samplingType samesequence --linearOutput --normalizeCPCScore --multiLevel --segmentationMode cosineDissimilarity --rnnModeSegment none
```

* mACPC:

```bash 
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION
--pathCheckpoint $PATH_CHECKPOINT_DIR --normMode layerNorm --dropout --n_process_loader 1 --batchSizeGPU 32 --CPCCTC --nPredicts 6 --CPCCTCNumMatched 12 --limitNegsInBatch 8 --nEpoch 50 --nGPU 2 --nLevelsGRU 2 --schedulerRamp 10 --normalizeCPCScore --multiLevel --segmentationMode cosineDissimilarity --nPredictsSegment 2 --CPCCTCNumMatchedSegment 4
```

* As in [Variable-rate hierarchical CPC leads to acoustic unit discovery in speech](https://arxiv.org/abs/2206.02211):

```bash 
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION
--pathCheckpoint $PATH_CHECKPOINT_DIR --normMode layerNorm --dropout --n_process_loader 1 --batchSizeGPU 32 --CPCCTC --nPredicts 6 --CPCCTCNumMatched 12 --limitNegsInBatch 8 --nEpoch 50 --nGPU 2 --nLevelsGRU 2 --schedulerRamp 10 --multiLevel --segmentationMode boundaryPredictor --nPredictsSegment 2 --adjacentNegatives --targetQuantizerSegment robustKmeans
```

Where:
- $PATH_AUDIO_FILES is the directory containing the audio files. The files should be arranged as below:
```
PATH_AUDIO_FILES  
│
└───speaker1
│   └───...
│         │   seq_11.{$EXTENSION}
│         │   seq_12.{$EXTENSION}
│         │   ...
│   
└───speaker2
    └───...
          │   seq_21.{$EXTENSION}
          │   seq_22.{$EXTENSION}
```

Please note that each speaker directory can contain an arbitrary number of subdirectories: the speaker label will always be retrieved from the top one. The name of the files isn't relevant. For a concrete example, you can look at the organization of the [Librispeech](http://www.openslr.org/12/) dataset.

- $PATH_CHECKPOINT_DIR in the directory where the checkpoints will be saved
- $TRAINING_SET is a path to a .txt file containing the list of the training sequences (see [here](https://drive.google.com/drive/folders/1BhJ2umKH3whguxMwifaKtSra0TgAbtfb) for example)
- $VALIDATION_SET is a path to a .txt file containing the list of the validation sequences
- $EXTENSION is the extension of each audio file

## Custom architectures

The code allows you to train a wide range of architectures. For example, to train the CPC method as described in [Van Den Oord's paper](https://arxiv.org/abs/1807.03748) just run:

```bash
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --normMode batchNorm --rnnMode linear
```

Or if you want to train a model with the architecture described in [Kreuk et al's paper](https://arxiv.org/abs/2007.13465):

```bash 
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION
--pathCheckpoint $PATH_CHECKPOINT_DIR --normMode layerNorm --dropout --n_process_loader 1 --batchSizeGPU 32 --nPredicts 1 --limitNegsInBatch 8 --nEpoch 50 --nGPU 2 --schedulerRamp 10 --rnnMode none --arMode no_ar --negativeSamplingExt 1 --nPredicts 1 --samplingType samesequence --linearOutput --normalizeCPCScore
```

Launch cpc/train.py -h to see all the possible options.

## How to restart a session

To restart a session from the last saved checkpoint just run
```bash
python cpc/train.py --pathCheckpoint $PATH_CHECKPOINT_DIR
```
## How to run an evaluation session

All evaluation scripts can be found in cpc/eval/.

### Linear separability:

After training, the CPC model can output high level features for a variety of tasks. For an input audio file sampled at 16kHz, the provided baseline model will output 256 dimensional output features every 10ms. We provide two linear separability tests one for speaker, one for phonemes, in which a linear classifier is trained on top of the CPC features with aligned labels, and evaluated on a held-out test set.

Train / Val splits as well as phone alignments for librispeech-100h can be found [here](https://drive.google.com/drive/folders/1BhJ2umKH3whguxMwifaKtSra0TgAbtfb).


Speaker separability:

```bash
python cpc/eval/linear_separability.py $PATH_DB $TRAINING_SET $VAL_SET $CHECKPOINT_TO_LOAD --pathCheckpoint $PATH_CHECKPOINT
```

Phone separability:
```bash
python cpc/eval/linear_separability.py $PATH_DB $TRAINING_SET $VAL_SET $CHECKPOINT_TO_LOAD --pathCheckpoint $PATH_CHECKPOINT --pathPhone $PATH_TO_PHONE_LABELS
```

PER using CTC loss and a non linear CNN+LSTM classifier:
```bash
python cpc/eval/linear_separability.py $PATH_DB $TRAINING_SET $VAL_SET $CHECKPOINT_TO_LOAD --pathCheckpoint $PATH_CHECKPOINT --pathPhone $PATH_TO_PHONE_LABELS --CTC --CTC_forbid_blank --useLSTM --convClassifier
```

### Phone segmentation:

You can run the segmentation evaluations metrics described in [Kreuk et al's paper](https://arxiv.org/abs/2007.13465) for a dataset for which you have phone alignments as the ones described for LibriSpeech above.

You can run a phone segmentation evaluation on a given checkpoint with:

```bash
python cpc/eval/segmentation.py --pathDB $PATH_AUDIO_FILES --load $PATH_CHECKPOINT --pathPhone $PATH_TO_PHONE_LABELS --file_extension $EXTENSION --pathCheckpoint $SAVE_DIR
```

## License

CPC_audio is MIT licensed, as found in the LICENSE file.
