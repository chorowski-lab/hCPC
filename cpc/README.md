# Repository's architecture

train.py : main script

dataset.py : defintion of the Librispeech dataset format

model.py : Basic encoders and AR models

feature_loader.py: different tools to load and save a CPC model.

transformers.py: an implementation of transformers

unit_tests.py : unit tests

criterion/: definition of the training criterions. Three criterion are currently available: CPC (unsupervised), speaker classification and phone classification.

eval/: evaluation scripts.

utils/: system utilities and misc.


## Stats module (initial) description

Under `stats` there are utils for computing stats. `stats/repr_diff_stat.py` is an example, `stats/stats_collector.py` is used to aggregate stats given as arguments to `train.py` and therefore each stat needs to be registered in `stats/stats_utils.py` similarly as `reprDiffStat` (`stats/repr_diff_stat.py`) is. 

To compute stats for `train.py` run, use `--captureSetStats` which needs to be passed in format `stat1Name:arg1,arg2,arg3_stat2Name:arg1,arg2` where args are stat-specific (example: `reprDiff:cosine,ctx_repr,0.05,../reprDiffHistograms`).

When specified like that (with `--captureSetStats`), stats are computed for "capture dataset" along with data capturing each specified number of epochs. One can specify to compute only stats and not capture data, but then captureDS still needs to be configured as described below under "CPC-CTC data capturing description". Example how to specify capture dataset: `--pathCaptureDS /pio/scratch/1/i283340/MGR/zs/sometries/ds2part.txt --captureEachEpochs 2`.


## Linear separability automation description:

This can be combined with data capturing described in the section below

There are some args added to train.py, in group_supervised_data and group_supervised_metric. In case I forget something here they also have some description there.
- --supervised_classif_metric is the flag to specify that additional linear separability task should be performed. Additionally, one/both of --speaker_sep, --path_phone_data should be specified to indicate which linear separabilities to perform - --speaker_sep for speaker classification and --path_phone_data for phoneme classification (this should be the path to the .txt file with phone alignments in their format, which they mention in the main readme of the repo)
- linear separability task can be run in two modes, either once on the trained checkpoint (--only_classif_metric) or each --linsep_classif_each_epochs epochs during main CPC training. To automatically perform linsep once each training, e.g. --linsep_classif_each_epochs 180 can be specified (now linsep is not done at 0. epoch)
- path where to store logs from linear separability task needs to be specified with --linsep_logs_dir; additionally, logging freqeuncy in epochs can be specified with --linsep_task_logging_step and those will be save under \<--linsep_logs_dir\>/\<CPC_epoch\>/phone  or \<--linsep_logs_dir\>/\<CPC_epoch\>/speaker
- path where to save classification models (state from best epoch for each separate classification training performed after X epoch of CPC training) can be specified with --linsep_checkpoint_dir and those will be save under \<linsep_checkpoint_dir\>/\<CPC_epoch\>/phone  or \<linsep_checkpoint_dir\>/\<CPC_epoch\>/speaker
- number of epochs to run each linear separability task for can be specified with --linsep_n_epoch
- additional linear separability task parameters can be specified with:
    - params to set for Adam optimizer: --linsep_lr , --linsep_beta1 , --linsep_beta2 , --linsep_epsilon
    - --phone_get_encoded to use CNN encodings for classification instead of produced contexts (this is only for phoneme classification with regular loss, other of their classifiers don’t support it so it doesn’t affect them; specifying this with classification CTC loss (below) is not supported and will yield assertion error)
    - --CTCphones to use CTC-based loss for classification instead of ‘regular’ loss assuming representations/contexts should be aligned with audio data
    - --linsep_net_layers to use bigger fully connected net during classification training (default: 1 - then there is just one matrix without activations; each layer has classification_class_number neurons except for CTC-based loss which has additional 1 (in last layer for blank symbol))
    - --linsepBatchSizeGPU can be specified to choose batch size for linear separability task; this is separate from batch size for CPC training

example run combined with data capturing:
some real training + capture
```
python train.py --pathDB /pio/data/zerospeech2021/LibriSpeech/train-clean-100 \
--pathTrain /pio/scratch/2/jch/wav2vec/LibriSpeech100_labels_split/train_split.txt \
--pathVal /pio/scratch/2/jch/wav2vec/LibriSpeech100_labels_split/test_split.txt \
--supervised_classif_metric \
--speaker_sep --path_phone_data /pio/scratch/1/i283340/MGR/zs/phones/converted_aligned_phones.txt \
--linsepBatchSizeGPU 32 --linsep_n_epoch 12 \
--linsep_logs_dir /pio/scratch/1/i283340/MGR/zs/linsep/logs2-001 \
--linsep_checkpoint_dir /pio/scratch/1/i283340/MGR/zs/linsep/checkp2-001 \
--linsep_classif_each_epochs 10 \
--pathCaptureDS /pio/scratch/1/i283340/MGR/zs/sometries/ds2part.txt \
--captureDStotNr 100 --captureEachEpochs 10 \
--pathCaptureSave /pio/scratch/1/i283340/MGR/zs/capture/try2-001 \
--captureConvRepr --captureCtxRepr --captureSpeakerAlign --capturePhoneAlign --capturePred --captureCPCCTCalign --captureCPCCTClogScores \
--pathCheckpoint /pio/scratch/1/i283340/MGR/zs/checkpoints/cpcctc_tests2-001 \
--file_extension .flac --n_process_loader 1 --max_size_loaded 40000000 \
--batchSizeGPU 16 --nPredicts 8 --CPCCTC --CPCCTCNumMatched 12 \
--CPCCTCSelfLoop --CPCCTCSkipBeg 1 --CPCCTCSkipEnd 2
```


## CPC-CTC data capturing description:

There are some new args added to group_save and group_db in train.py for capturing - options are also described in the ‘help’ argument in definitions in case I forget something here:
- Data capturing is possible in 2 modes: capture once for a teached model (use --onlyCapture and --pathCheckpoint) or capture each N epochs during training each N epochs (don’t use --onlyCapture and only specify --captureEachEpochs and things from two bullets below)
- The data is captured for a separately specified dataset I call captureDataset. This can e.g. be just same as valDataset. It is specified with --pathCaptureDS that is the path to .txt file with sequences in this DS. Additionally, --captureDSfreq OR --captureDStotNr can be used to sample only the part of sequences specified in the file - some percentage of those with freq one, and total number with totNr one. (example: --pathCaptureDS \<valDSpath\> --captureDStotNr 8 can be used to capture for just 8 audio files of the val dataset)
- --pathCaptureSave tells where to save the data. Data for each epoch (for 1 epoch if --onlyCapture) is saved under \<that_dir\>/\<num_epoch\>/\<what_is_captured\>/ with file names {what_is_captured}_batch{batchBegin}-{batchEnd}.pt in one file each thing for each batch (example: ctx_repr_batch0-15.pt under ./captureRoot/0/ctx_repr/). What to capture is chosen with --captureConvRepr , --captureCtxRepr, --captureSpeakerAlign, --capturePhoneAlign, --capturePred , --captureCPCCTCalign , --captureCPCCTClogScores args (so, those are: representations, LSTM-produced contexts, speaker alignments for the audio, phoneme alignments for the audio, CPC predictions, CPC-CTC alignments). Note that capturing speaker and phoneme alignments is necessary for their visualization, as it is later impossible to tell from what audio file particular batch was taken (audio files are glued together and chunked, and also randomly permuted). There is also --captureEverything added for convenience that captures everything that is valid for given run config, but it’s alway safer to specify exactly what to capture. For capturing phoneme alignments --path_phone_data needs to be specified (this is the path to a .txt file with phoneme alignments in their format, they provide it somewhere in repo’s main readme)

IN CASE YOU RUN DATA CAPTURE FOR AN ALREADY TRAINED MODEL, PASS SAME ARGUMENTS FOR THE MODEL TO LOAD CORRECTLY

Example run that saves data each 2 epochs for 8 audio files of val dataset (with some very small dummy train=val datasets I made):
```
python train.py --pathDB /pio/scratch/1/i283340/MGR/zs/ds2
--pathTrain /pio/scratch/1/i283340/MGR/zs/sometries/ds2part.txt
--pathVal /pio/scratch/1/i283340/MGR/zs/sometries/ds2part.txt
--pathCaptureDS /pio/scratch/1/i283340/MGR/zs/sometries/ds2part.txt
--captureDStotNr 8 --captureEachEpochs 2
--pathCaptureSave /pio/scratch/1/i283340/MGR/zs/capture/try1
--path_phone_data /pio/scratch/1/i283340/MGR/zs/phones/converted_aligned_phones.txt
--captureConvRepr --captureCtxRepr --captureSpeakerAlign --capturePhoneAlign --capturePred --captureCPCCTCalign --captureCPCCTClogScores
--pathCheckpoint /pio/scratch/1/i283340/MGR/zs/checkpoints/cpcctc_tests2
--file_extension .flac --n_process_loader 2 --max_size_loaded 40000000
--batchSizeGPU 16 --nPredicts 8 --CPCCTC --CPCCTCNumMatched 12
--CPCCTCSelfLoop --CPCCTCSkipBeg 1 --CPCCTCSkipEnd 2
```

Example with just capturing:
```
python train.py --pathDB /pio/data/zerospeech2021/LibriSpeech/train-clean-100 --onlyCapture \
--pathCaptureDS /pio/scratch/2/jch/wav2vec/LibriSpeech100_labels_split/test_split.txt \
--captureDStotNr 100 \
--pathCaptureSave /pio/gluster/i283340/cpccapture/ls100_cpcctc_match12_pred8/ \
--captureConvRepr --captureCtxRepr --captureSpeakerAlign --capturePhoneAlign --capturePred --captureCPCCTCalign --captureCPCCTClogScores \
--path_phone_data /pio/scratch/1/i283340/MGR/zs/phones/converted_aligned_phones.txt \
--pathCheckpoint /pio/gluster/i283340/modelcpy/ls100_cpcctc_match12_pred8 \
--file_extension .flac \
--normMode layerNorm --dropout --rnnMode transformer --n_process_loader 1 --max_size_loaded 4000000000 --nLevelsGRU 2 \
--batchSizeGPU 32 --limitNegsInBatch 8 --schedulerRamp 10 --nPredicts 8 --CPCCTC --CPCCTCNumMatched 12
```