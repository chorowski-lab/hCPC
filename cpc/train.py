# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import os
from turtle import down
import numpy as np
import torch
import time
from copy import deepcopy
import random
import psutil
import sys
#import torchaudio

import cpc.criterion as cr
import cpc.criterion.soft_align as sa
import cpc.model as model
import cpc.utils.misc as utils
import cpc.feature_loader as fl
import cpc.eval.linear_separability as linsep
from cpc.cpc_default_config import set_default_cpc_config
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels
import cpc.stats.stat_utils as statutil
import itertools
from torch.utils.tensorboard import SummaryWriter
from cpc.utils.misc import Globals

def getCriterion(args, downsampling, nSpeakers, nPhones):
    dimFeatures = args.hiddenGar if not args.onEncoder else args.hiddenEncoder
    if not args.supervised:
        if args.cpc_mode == 'none':
            cpcCriterion = cr.NoneCriterion()
        else:
            useFrameLvlCriterion = not args.freezeFrameModel and not args.segmentLevel
            sizeInputSeq = (args.sizeWindow // downsampling)
            if args.CPCCTC:
                targetQuantizer = None
                if args.encodingsQuantizer == 'none':
                    if args.targetQuantizer == 'gumbel':
                        targetQuantizer = model.GumbelQuantizer(args.hiddenEncoder, args.hiddenEncoder, 
                        args.numGroupsCodebook, args.numCodesCodebook)
                    elif args.targetQuantizer == 'kmeans':
                        targetQuantizer = model.KMeansQuantizer(args.hiddenEncoder, args.hiddenEncoder,
                        args.numGroupsCodebook, args.numCodesCodebook)
                    elif args.contextQuantizer == 'robustKmeans':
                        targetQuantizer = model.RobustKMeansQuantizer(args.hiddenEncoder, args.hiddenEncoder, args.numCodesCodebook)
                cpcCriterion = None
                if useFrameLvlCriterion:
                    cpcCriterion = sa.CPCUnsupersivedCriterion(args.nPredicts,
                                                            args.CPCCTCNumMatched,
                                                            args.hiddenGar,
                                                            args.hiddenEncoder,
                                                            args.negativeSamplingExt,
                                                            allowed_skips_beg=args.CPCCTCSkipBeg,
                                                            allowed_skips_end=args.CPCCTCSkipEnd,
                                                            predict_self_loop=args.CPCCTCSelfLoop,
                                                            learn_blank=args.CPCCTCLearnBlank,
                                                            normalize_enc=args.CPCCTCNormalizeEncs,
                                                            normalize_preds=args.CPCCTCNormalizePreds,
                                                            masq_rules=args.CPCCTCMasq,
                                                            no_negs_in_match_window=args.CPCCTCNoNegsMatchWin,
                                                            loss_temp=args.CPCCTCLossTemp,
                                                            limit_negs_in_batch=args.limitNegsInBatch,
                                                            mode=args.cpc_mode,
                                                            rnnMode=args.rnnMode,
                                                            dropout=args.dropout,
                                                            speakerEmbedding=args.speakerEmbedding,
                                                            nSpeakers=nSpeakers,
                                                            sizeInputSeq=sizeInputSeq,
                                                            normalizeScore=args.normalizeCPCScore,
                                                            targetQuantizer=targetQuantizer)
                if args.multiLevel or args.segmentLevel:
                    targetQuantizerSegment = None
                    if args.targetQuantizerSegment == 'gumbel':
                        targetQuantizerSegment = model.GumbelQuantizer(256, 256, args.numGroupsCodebook, args.numCodesCodebook)
                    elif args.targetQuantizerSegment == 'kmeans':
                        targetQuantizerSegment = model.KMeansQuantizer(256, 256, args.numGroupsCodebook, args.numCodesCodebook)
                    elif args.targetQuantizerSegment == 'robustKmeans':
                        targetQuantizerSegment = model.RobustKMeansQuantizer(256, 256, args.numCodesCodebook)
                    cpcCriterion = sa.MultiLevelCriterion(cpcCriterion, 
                                                          args.nPredictsSegment, 
                                                          args.CPCCTCNumMatchedSegment,
                                                          args.hiddenGarSegment,
                                                          args.hiddenEncoderSegment,
                                                          args.negativeSamplingExtSegment,
                                                          sizeInputSeq,
                                                          args.normalizeCPCScore,
                                                          targetQuantizerSegment,
                                                          args.adjacentNegatives,
                                                          args.rlSetup)
            else:
                cpcCriterion = cr.CPCUnsupersivedCriterion(args.nPredicts,
                                                        args.hiddenGar,
                                                        args.hiddenEncoder,
                                                        args.negativeSamplingExt,
                                                        normalizeScore=args.normalizeCPCScore,
                                                        mode=args.cpc_mode,
                                                        rnnMode=args.rnnMode,
                                                        dropout=args.dropout,
                                                        nSpeakers=nSpeakers,
                                                        speakerEmbedding=args.speakerEmbedding,
                                                        sizeInputSeq=sizeInputSeq)
    elif args.pathPhone is not None:
        if not args.CTC:
            cpcCriterion = cr.PhoneCriterion(dimFeatures,
                                             nPhones, args.onEncoder,
                                             nLayers=args.nLevelsPhone)
        else:
            cpcCriterion = cr.CTCPhoneCriterion(dimFeatures,
                                                nPhones, args.onEncoder)
    else:
        cpcCriterion = cr.SpeakerCriterion(dimFeatures, nSpeakers)
    return cpcCriterion


def loadCriterion(pathCheckpoint, downsampling, nSpeakers, nPhones):
    _, _, locArgs = fl.getCheckpointData(os.path.dirname(pathCheckpoint))
    criterion = getCriterion(locArgs, downsampling, nSpeakers, nPhones)

    state_dict = torch.load(pathCheckpoint, 'cpu')

    criterion.load_state_dict(state_dict["cpcCriterion"])
    return criterion


def trainStep(dataLoader,
              cpcModel,
              cpcCriterion,
              optimizer,
              scheduler,
              loggingStep,
              headWeights,
              PhoneLabels,
              auxCriterion,
              predictClusterIdCriterion,
              trainPolicy):

    cpcModel.train()
    cpcCriterion.train()    

    start_time = time.perf_counter()
    n_examples = 0
    logs, lastlogs = {}, None
    iter = 0
    for step, fulldata in enumerate(dataLoader):
        Globals.currentIteration += 1
        batchData, labelData = fulldata
        label = labelData['speaker'] if PhoneLabels is None else labelData['phone']
        n_examples += batchData.size(0)
        batchData = batchData.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        modelOutput = cpcModel(batchData, label)
        # if modelOutput is None:
        #     continue
        c_feature, encoded_data, label, extraLossesModel = modelOutput
        allLosses, allAcc, captureRes, extraLossesCriterion = cpcCriterion(c_feature, encoded_data, label, ['seqLosses'] if fl.get_module(cpcModel).numUpdates % loggingStep == 0 else None)
        totLoss = 0
        # if not trainPolicy:
        for i, loss in enumerate(allLosses):
            totLoss += headWeights[i] * loss.mean()
        totLoss = totLoss / len(allLosses)

        for lossTag, lossValue in itertools.chain(extraLossesModel.items(), extraLossesCriterion.items()):
            lossValue = lossValue.mean()
            if f"{str(lossTag)}_train" not in logs:
                logs[f"{str(lossTag)}_train"] = np.zeros((1,))
            logs[f"{str(lossTag)}_train"] += lossValue.detach().cpu().numpy()
            # if lossTag in ['piGradLoss', 'PhoneDensityPriorLoss']:
            #     if trainPolicy:
            #         totLoss += lossValue
            # elif not trainPolicy:
            totLoss += lossValue
            
        if auxCriterion is not None:
            allLossesAux, allAccAux, _ = auxCriterion([c_feature[0]], c_feature[0], label, None)
            if f"kreukLoss_train" not in logs:
                logs[f"kreukLoss_train"] = np.zeros(allLossesAux[0].size(1))
                logs[f"kreukAcc_train"] = np.zeros(allLossesAux[0].size(1))
            logs[f"kreukLoss_train"] += (allLossesAux[0].mean(dim=0)).detach().cpu().numpy()
            logs[f"kreukAcc_train"] += (allAccAux[0].mean(dim=0)).cpu().numpy()
            totLoss += allLossesAux[0].mean()
        if predictClusterIdCriterion is not None:
            allLossesClusterId, allAccClusterId = predictClusterIdCriterion(c_feature[0], encoded_data, label, None)
            if f"ClusterIdLoss_train" not in logs:
                logs[f"ClusterIdLoss_train"] = np.zeros(allLossesClusterId.size(1))
                logs[f"ClusterIdAcc_train"] = np.zeros(allAccClusterId.size(1))
            logs[f"ClusterIdLoss_train"] += np.asarray([allLossesClusterId.mean().item()])
            logs[f"ClusterIdAcc_train"] += np.asarray([allAccClusterId.mean().item()])
            totLoss += allLossesClusterId.mean()
        totLoss.backward()
        # writer.add_histogram(tag='SeqLosses', values=captureRes[0]['seqLosses'], global_step=fl.get_module(cpcModel).numUpdates)
        if fl.get_module(cpcModel).numUpdates % loggingStep == 0 and not Globals.debugging:
            for l in range(len(captureRes)):
                for n, v in captureRes[l].items():
                    Globals.writer.add_histogram(tag=n, values=v, global_step=fl.get_module(cpcModel).numUpdates)
                    
            for n, p in itertools.chain(fl.get_module(cpcModel).named_parameters(), fl.get_module(cpcCriterion).named_parameters()):
                if p.requires_grad:
                    try:
                        Globals.writer.add_histogram(tag='Parameters/' + n, values=p, global_step=fl.get_module(cpcModel).numUpdates)
                        Globals.writer.add_scalar(tag='Gradient norms/' + n, scalar_value=torch.linalg.norm(p.grad), global_step=fl.get_module(cpcModel).numUpdates)
                    except:
                        pass

        # if isinstance(fl.get_module(cpcModel), model.MultiLevelModel):
        #     segmenter = fl.get_module(cpcModel).segmenter
        #     if segmenter is not None:
        #         torch.nn.utils.clip_grad_norm_(segmenter.parameters(), 0.1)
        # Show grads ?
        optimizer.step()
        optimizer.zero_grad()

        fl.get_module(cpcModel).updateCounter()
        fl.get_module(cpcCriterion).updateCounter()

        for headId in range(len(allLosses)):   
            if f"locLoss_train_head{headId}" not in logs:
                logs[f"locLoss_train_head{headId}"] = np.zeros(allLosses[headId].size(1))
                logs[f"locAcc_train_head{headId}"] = np.zeros(allLosses[headId].size(1))

            logs[f"locLoss_train_head{headId}"] += (allLosses[headId].mean(dim=0)).detach().cpu().numpy()
            logs[f"locAcc_train_head{headId}"] += (allAcc[headId].mean(dim=0)).cpu().numpy()
        iter += 1

        if (step + 1) % loggingStep == 0:
            new_time = time.perf_counter()
            elapsed = new_time - start_time
            print(f"Update {step + 1}")
            print(f"elapsed: {elapsed:.1f} s")
            print(
                f"{1000.0 * elapsed / loggingStep:.1f} ms per batch, {1000.0 * elapsed / n_examples:.1f} ms / example")
            locLogs = utils.update_logs(logs, loggingStep, lastlogs)
            lastlogs = deepcopy(logs)
            utils.show_logs("Training loss", locLogs)
            start_time, n_examples = new_time, 0

    if scheduler is not None:
        scheduler.step()

    logs = utils.update_logs(logs, iter)
    logs["iter"] = iter
    utils.show_logs("Average training loss on epoch", logs)
    return logs


def valStep(dataLoader,
            cpcModel,
            cpcCriterion,
            PhoneLabels,
            auxCriterion,
            predictClusterIdCriterion):

    cpcCriterion.eval()
    cpcModel.eval()
    logs = {}
    cpcCriterion.eval()
    cpcModel.eval()
    iter = 0

    for step, fulldata in enumerate(dataLoader):

        batchData, labelData = fulldata
        label = labelData['speaker'] if PhoneLabels is None else labelData['phone']

        batchData = batchData.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        with torch.no_grad():
            modelOutput = cpcModel(batchData, label)
            # if modelOutput is None:
            #     continue
            c_feature, encoded_data, label, extraLossesModel = modelOutput        
            allLosses, allAcc, _, extraLossesCriterion = cpcCriterion(c_feature, encoded_data, label, None)
        
        for lossTag, lossValue in itertools.chain(extraLossesModel.items(), extraLossesCriterion.items()):
            lossValue = lossValue.mean()
            if f"{str(lossTag)}_val" not in logs:
                logs[f"{str(lossTag)}_val"] = np.zeros((1,))
            logs[f"{str(lossTag)}_val"] += lossValue.cpu().numpy()

        for headId in range(len(allLosses)):
            if f"locLoss_val_head{headId}" not in logs:
                logs[f"locLoss_val_head{headId}"] = np.zeros(allLosses[headId].size(1))
                logs[f"locAcc_val_head{headId}"] = np.zeros(allLosses[headId].size(1))

            logs[f"locLoss_val_head{headId}"] += allLosses[headId].mean(dim=0).cpu().numpy()
            logs[f"locAcc_val_head{headId}"] += allAcc[headId].mean(dim=0).cpu().numpy()
        iter += 1

    logs = utils.update_logs(logs, iter)
    logs["iter"] = iter
    utils.show_logs("Validation loss:", logs)
    return logs


def captureStep(
            dataLoader,
            cpcModel,
            cpcCriterion,
            captureOptions,
            captureStatsCollector,
            epochNr):

    cpcCriterion.eval()
    cpcModel.eval()
    logs = {}
    cpcCriterion.eval()
    cpcModel.eval()
    iter = 0

    capturePath = captureOptions['path']
    whatToSave = captureOptions['what']
    cpcCaptureOpts = []
    if 'pred' in whatToSave:
        cpcCaptureOpts.append('pred')
    if 'cpcctc_align' in whatToSave:
        cpcCaptureOpts.append('cpcctc_align')
    if 'cpcctc_log_scores' in whatToSave:
        cpcCaptureOpts.append('cpcctc_log_scores')

    # they merge (perhaps each speaker's) audio into one long chunk
    # and AFAIU sample can begin in one file and end in other one
    # so won't try to mess up with tracking filenames, saving samples just as 1, 2, etc.

    if captureStatsCollector:
        captureStatsCollector.zeroStats()

    batchBegin = 0
    epochDir = os.path.join(capturePath, str(epochNr))
    if not os.path.exists(epochDir):
        os.makedirs(epochDir)
    for sub in whatToSave:
        if not os.path.exists(os.path.join(epochDir, sub)):
            os.makedirs(os.path.join(epochDir, sub))

    for step, fulldata in enumerate(dataLoader):

        batchData, labelData = fulldata
        labelSpeaker = labelData['speaker']
        batchEnd = batchBegin + batchData.shape[0] - 1

        batchData = batchData.cuda(non_blocking=True)
        labelSpeaker = labelSpeaker.cuda(non_blocking=True)

        with torch.no_grad():

            c_feature, encoded_data, labelSpeaker = cpcModel(batchData, labelSpeaker)
            allLosses, allAcc, captured = cpcCriterion(c_feature, encoded_data, labelSpeaker, cpcCaptureOpts)
        
            # saving it with IDs like that assumes deterministic order of elements
            # which is there as dataLoader is a sequential one here
            if 'conv_repr' in whatToSave:
                # encoded data shape: batch_size x len x repr_dim
                torch.save(encoded_data.cpu(), os.path.join(epochDir, 'conv_repr', f'conv_repr_batch{batchBegin}-{batchEnd}.pt'))
            if 'ctx_repr' in whatToSave:
                # ctx data shape: also batch_size x len x repr_dim
                torch.save(c_feature.cpu(), os.path.join(epochDir, 'ctx_repr', f'ctx_repr_batch{batchBegin}-{batchEnd}.pt'))
            if 'speaker_align' in whatToSave:
                # speaker data shape: batch_size (1-dim, each one in batch is whole by 1 speaker)
                torch.save(labelSpeaker.cpu(), os.path.join(epochDir, 'speaker_align', f'speaker_align_batch{batchBegin}-{batchEnd}.pt'))
            if 'phone_align' in whatToSave:
                # phone alignment data shape: batch_size x len
                torch.save(labelData['phone'].cpu(), os.path.join(epochDir, 'phone_align', f'phone_align_batch{batchBegin}-{batchEnd}.pt'))
            for cpcCaptureThing in cpcCaptureOpts:
                # pred shape (CPC-CTC): batch_size x (len - num_matched) x repr_dim x num_predicts (or num_predicts +1 if self loop allowed)
                # cpcctc_align shape (CPC-CTC): batch_size x (len - num_matched) x num_matched
                # cpcctc_log_scores shape (CPC-CTC): batch_size x (len - num_matched) x num_matched x num_predicts (or num_predicts +1 if self loop allowed)
                torch.save(captured[cpcCaptureThing].cpu(), os.path.join(epochDir, cpcCaptureThing, 
                            f'{cpcCaptureThing}_batch{batchBegin}-{batchEnd}.pt'))

            if captureStatsCollector:
                allBatchData = {}
                allBatchData['conv_repr'] = encoded_data
                allBatchData['ctx_repr'] = c_feature
                allBatchData['speaker_align'] = labelSpeaker
                if 'phone' in labelData:
                    allBatchData['phone_align'] = labelData['phone']
                # ones below are only ones that need to be captured(saved) in order to be available for stats
                for cpcCaptureThing in captured:
                    allBatchData[cpcCaptureThing] = captured[cpcCaptureThing]

                captureStatsCollector.batchUpdate(allBatchData)

            # TODO maybe later can write that with process pool or something??? but not even sure if makes sense

        batchBegin += batchData.shape[0]

    if captureStatsCollector:
        captureStatsCollector.logStats(epochNr)
    return


def run(trainDataset,
        valDataset,
        captureDatasetWithOptions,
        linsepClassificationTaskConfig,
        batchSize,
        samplingMode,
        cpcModel,
        cpcCriterion,
        nEpoch,
        pathCheckpoint,
        optimizer,
        scheduler,
        logs,
        headWeights,
        PhoneLabels,
        auxCriterion=None,
        predictClusterIdCriterion=None):
    Globals.writer = SummaryWriter(f"runs/{pathCheckpoint.split('/')[-2]}")
    startEpoch = len(logs["epoch"])
    print(f"Running {nEpoch} epochs, now at {startEpoch}")
    bestAcc = 0
    currentAccuracy = 0
    bestStateDict = None
    # bestSegmenterStateDict = None
    start_time = time.time()
    
    captureDataset, captureOptions, captureStatsCollector = captureDatasetWithOptions
    linsepEachEpochs, linsepFun = linsepClassificationTaskConfig
    assert (captureDataset is None and captureOptions is None) \
        or (captureDataset is not None and captureOptions is not None)
    if captureOptions is not None:
        captureEachEpochs = captureOptions['eachEpochs']

    print(f'DS sizes: train {str(len(trainDataset)) if trainDataset is not None else "-"}, '
        f'val {str(len(valDataset)) if valDataset is not None else "-"}, capture '
        f'{str(len(captureDataset)) if captureDataset is not None else "-"}')

    for epoch in range(startEpoch, nEpoch):
        Globals.epoch = epoch
        print(f"Starting epoch {epoch}")
        utils.cpu_stats()

        trainLoader = trainDataset.getDataLoader(batchSize, samplingMode,
                                                True, numWorkers=0)
        
        valLoader = valDataset.getDataLoader(batchSize, 'sequential', False,
                                            numWorkers=0)
        
        if captureDataset is not None and epoch % captureEachEpochs == 0:
            captureLoader = captureDataset.getDataLoader(batchSize, 'sequential', False,
                                                numWorkers=0)
        
        print("Training dataset %d batches, Validation dataset %d batches, batch size %d" %
            (len(trainLoader), len(valLoader), batchSize))

        # beta = max(1.0 - 0.05 * epoch, 0.0)
        # headWeights[0] = beta
        # headWeights[1] = 1 - beta
        # print("Head weights: ", headWeights)
        locLogsTrain = trainStep(trainLoader, cpcModel, cpcCriterion,
                                optimizer, scheduler, logs["logging_step"], headWeights, 
                                PhoneLabels, auxCriterion, predictClusterIdCriterion, epoch % 2 == 0)

        locLogsVal = valStep(valLoader, cpcModel, cpcCriterion, PhoneLabels, auxCriterion, predictClusterIdCriterion)

        if captureDataset is not None and epoch % captureEachEpochs == 0:
            print(f"Capturing data for epoch {epoch}")
            captureStep(captureLoader, cpcModel, cpcCriterion, captureOptions, captureStatsCollector, epoch)

        # if cpcCriterion.smartPooling:
        #     currentAccuracy = (float(locLogsVal["locAcc_val_head0"].mean()) + float(locLogsVal["locAcc_val_head1"].mean())) / 2
        # else:
        if 'locAcc_val_head1' in locLogsVal:
            currentAccuracy = float(locLogsVal["locAcc_val_head1"].mean())
        if currentAccuracy > bestAcc:
            bestAcc = currentAccuracy
            bestStateDict = deepcopy(fl.get_module(cpcModel).state_dict())
            # bestSegmenterStateDict = deepcopy(fl.get_module(cpcModel).segmenter.state_dict())
        # elif Globals.epoch >= 10:
        #     print("Falling back to previous segmenter")
        #     fl.get_module(cpcModel).segmenter.load_state_dict(bestSegmenterStateDict, strict=True)

        locLogsLinsep = {}
        # this performs linsep task for the best CPC model up to date
        if linsepEachEpochs is not None and epoch !=0 and epoch % linsepEachEpochs == 0:
            # capturing for current CPC state after this epoch, relying on CPC internal accuracy is vague
            locLogsLinsep = linsepFun(epoch, cpcModel, epoch)

        print(f'Ran {epoch + 1} epochs '
            f'in {time.time() - start_time:.2f} seconds')

        torch.cuda.empty_cache()

        for key, value in dict(locLogsTrain, **locLogsVal, **locLogsLinsep).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            while len(logs[key]) < len(logs["epoch"]):
                logs[key].append(None)  # for not-every-epoch-logged things
            logs[key].append(value)

        logs["epoch"].append(epoch)

        if pathCheckpoint is not None \
                and (epoch % logs["saveStep"] == 0 or epoch == nEpoch-1):

            modelStateDict = fl.get_module(cpcModel).state_dict()
            criterionStateDict = fl.get_module(cpcCriterion).state_dict()

            fl.save_checkpoint(modelStateDict, criterionStateDict,
                            optimizer.state_dict(), bestStateDict,
                            f"{pathCheckpoint}_{epoch}.pt")
            utils.save_logs(logs, pathCheckpoint + "_logs.json")
    Globals.writer.close()


def onlyCapture(
        captureDatasetWithOptions,
        batchSize,
        cpcModel,
        cpcCriterion,
        logs
):
    startEpoch = len(logs["epoch"])
    captureDataset, captureOptions, captureStatsCollector = captureDatasetWithOptions
    assert (captureDataset is not None and captureOptions is not None)
    if captureOptions is not None:
        captureEachEpochs = captureOptions['eachEpochs']
    print(f'Capture DS size: {str(len(captureDataset))}')

    captureLoader = captureDataset.getDataLoader(batchSize, 'sequential', False,
                                                numWorkers=0)
    print(f"Capturing data for model checkpoint after epoch: {startEpoch-1}")
    captureStep(captureLoader, cpcModel, cpcCriterion, captureOptions, captureStatsCollector, startEpoch-1)


def main(args):
    args = parseArgs(args)

    if args.debug:
        Globals.debugging = True
        import ptvsd
        ptvsd.enable_attach(('0.0.0.0', 7310))
        print("Attach debugger now")
        ptvsd.wait_for_attach()
        args.nGPU = 2

    utils.set_seed(args.random_seed)
    logs = {"epoch": [], "iter": [], "saveStep": args.save_step}
    loadOptimizer = False
    os.makedirs(args.pathCheckpoint, exist_ok=True)
    if not args.onlyCapture and not args.only_classif_metric:
        json.dump(vars(args), open(os.path.join(args.pathCheckpoint, 'checkpoint_args.json'), 'wt'))
    if args.pathCheckpoint is not None and not args.restart:
        cdata = fl.getCheckpointData(args.pathCheckpoint)
        if cdata is not None:
            data, logs, locArgs = cdata
            if args.restartEpochCount:
                logs = {"epoch": [], "iter": [], "saveStep": args.save_step}
            print(f"Checkpoint detected at {data}")
            fl.loadArgs(args, locArgs,
                        forbiddenAttr={"nGPU", "pathCheckpoint",
                                       "debug", "restart", "world_size",
                                       "n_nodes", "node_id", "n_gpu_per_node",
                                       "max_size_loaded"})
            args.load, loadOptimizer = [data], True
            args.loadCriterion = True

    logs["logging_step"] = args.logging_step

    print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=True)}')
    print('-' * 50)

    seqNames, speakers = findAllSeqs(args.pathDB,
                                     extension=args.file_extension,
                                     loadCache=not args.ignore_cache)

    if not args.onlyCapture or args.only_classif_metric:
        print(f'Found files: {len(seqNames)} seqs, {len(speakers)} speakers')
        # Datasets
        if args.pathTrain is not None and len(args.pathTrain) == len(args.pathDB):
            seqTrain = filterSeqs(args.pathTrain, seqNames)
        else:
            seqTrain = seqNames

        if args.pathVal is None:
            random.shuffle(seqTrain)
            sizeTrain = int(0.99 * len(seqTrain))
            seqTrain, seqVal = seqTrain[:sizeTrain], seqTrain[sizeTrain:]
            print(f'Found files: {len(seqTrain)} train, {len(seqVal)} val')
        else:
            seqVal = filterSeqs(args.pathVal, seqNames)

    if args.pathCaptureDS is not None:
        assert args.pathCaptureSave is not None
        whatToSave = []
        if args.captureEverything:
            whatToSave = ['conv_repr', 'ctx_repr', 'speaker_align', 'pred']
            if args.path_phone_data:
                whatToSave.append('phone_align')
            if args.CPCCTC:
                whatToSave.append('cpcctc_align')
                whatToSave.append('cpcctc_log_scores')
        else:
            for argVal, name in zip([args.captureConvRepr, 
                                    args.captureCtxRepr, 
                                    args.captureSpeakerAlign, 
                                    args.capturePhoneAlign,
                                    args.capturePred,
                                    args.captureCPCCTCalign,
                                    args.captureCPCCTClogScores], 
                                    ['conv_repr', 'ctx_repr', 'speaker_align', 'phone_align', 'pred', 'cpcctc_align', 'cpcctc_log_scores']):
                if argVal:
                    whatToSave.append(name)
        ###assert len(whatToSave) > 0
        captureOptions = {
            'path': args.pathCaptureSave,
            'eachEpochs': args.captureEachEpochs,
            'what': whatToSave
        }
        seqCapture = filterSeqs(args.pathCaptureDS, seqNames, 
                                percentage=args.captureDSfreq, totalNum=args.captureDStotNr)
        print(f'Capture files: {len(seqCapture)}')
    else:
        seqCapture = None
        captureOptions = None

    if not args.onlyCapture:
        if args.debug:
            seqTrain = seqTrain[-1000:]
            seqVal = seqVal[-100:]

        phoneLabels, nPhones = None, None
        # if args.supervised and args.pathPhone is not None:
        #     print("Loading the phone labels at " + args.pathPhone)
        #     phoneLabels, nPhones = parseSeqLabels(args.pathPhone)
        #     print(f"{nPhones} phones found")

        if args.path_phone_data is not None:
            print("Loading the phone labels at " + args.path_phone_data)
            phoneLabels, nPhones = parseSeqLabels(args.path_phone_data)
            print(f"{nPhones} phones found")

        print("")
        print(f'Loading audio data at {args.pathDB}')
        print("Loading the training dataset")
        trainDataset = AudioBatchData(args.pathDB,
                                    args.sizeWindow,
                                    seqTrain,
                                    phoneLabels,
                                    len(speakers),
                                    nProcessLoader=args.n_process_loader,
                                    MAX_SIZE_LOADED=args.max_size_loaded)
        print("Training dataset loaded")
        print("")

        print("Loading the validation dataset")
        valDataset = AudioBatchData(args.pathDB,
                                    args.sizeWindow,
                                    seqVal,
                                    phoneLabels,
                                    len(speakers),
                                    nProcessLoader=args.n_process_loader)
        print("Validation dataset loaded")
        print("")
    else:
        phoneLabels, nPhones = None, None
        trainDataset = None
        valDataset = None

    if seqCapture is not None:

        if args.path_phone_data:
            print("Loading the phone labels at " + args.path_phone_data)
            phoneLabelsForCapture, _ = parseSeqLabels(args.path_phone_data)
        else:
            assert not args.capturePhoneAlign
            phoneLabelsForCapture = None
            
        print("Loading the capture dataset")
        captureDataset = AudioBatchData(args.pathDB,
                                    args.sizeWindow,
                                    seqCapture,
                                    phoneLabelsForCapture,
                                    len(speakers),
                                    nProcessLoader=args.n_process_loader)
        print("Capture dataset loaded")
        print("")

        if args.captureSetStats:
            captureSetStatsCollector = statutil.constructStatCollectorFromSpecs(args.captureSetStats)
        else:
            captureSetStatsCollector = None
    else:
        captureDataset = None
        captureSetStatsCollector = None

    if args.load is not None:
        if args.gru_level is not None and args.gru_level > 0:
            updateConfig = argparse.Namespace(nLevelsGRU=args.gru_level)
        else:
            updateConfig = None


        # loadBestNotLast = args.onlyCapture or args.only_classif_metric
        # could use this option for loading best state when not running actual training
        # but relying on CPC internal acc isn't very reliable
        # [!] caution - because of how they capture checkpoints,
        #     they capture "best in this part of training" as "best" (apart from capturing current state)
        #     so if best is in epoch 100 and training is paused and resumed from checkpoint
        #     in epoch 150, checkpoint from epoch 200 has "best from epoch 150" saved as globally best
        #     (but this is internal-CPC-score best anyway, which is quite vague)
        cpcModel, args.hiddenGar, args.hiddenEncoder = \
            fl.loadModel(args.load, load_nullspace=args.nullspace, updateConfig=updateConfig, loadOnlyFrameModel=args.loadOnlyFrameModel)

        if (args.multiLevel or args.segmentLevel) and isinstance(cpcModel, model.CPCModel):
            print("Appending segment-level network to loaded Model")
            segmenter = None
            if not args.noSegmentation:
                segmenter = model.Segmenter(args.segmentationMode, 
                                            args.segmentOnContext, 
                                            args.nPredictsSegment + 2 if args.adjacentNegatives else args.CPCCTCNumMatchedSegment + 1,
                                            args.segmentCompression,
                                            args.hiddenGar if args.segmentOnContext else args.hiddenEncoder,
                                            args.nLayersBoundaryPredictor)
            cpcModel = model.MultiLevelModel(cpcModel, segmenter, keepHidden=args.samplingType == "sequential")

        if args.gru_level is not None and args.gru_level > 0 and args.samplingType == "sequential":
            # Keep hidden units at LSTM layers on sequential batches
            if args.nullspace:
                cpcModel.cpc.gAR.keepHidden = True
            else:
                if isinstance(cpcModel, model.MultiLevelModel):
                    cpcModel.frameLevelModel.gAR.keepHidden = True
                else:
                    cpcModel.gAR.keepHidden = True
    else:
        # Encoder network
        encoderNet = fl.getEncoder(args)
        # AR Network        
        arNet = fl.getAR(args)
        # Quantizers
        quantizerEncodings, quantizerContext = fl.getQuantizers(args)
        cpcModel = model.CPCModel(encoderNet, arNet, quantizerEncodings, quantizerContext)
        if args.multiLevel or args.segmentLevel:
            segmenter = None
            if not args.noSegmentation:
                segmenter = model.Segmenter(args.segmentationMode, 
                                            args.segmentOnContext, 
                                            args.nPredictsSegment + 2 if args.adjacentNegatives else args.CPCCTCNumMatchedSegment + 1,
                                            args.segmentCompression,
                                            args.hiddenGar if args.segmentOnContext else args.hiddenEncoder,
                                            args.nLayersBoundaryPredictor)
            cpcModel = model.MultiLevelModel(cpcModel, segmenter, keepHidden=args.samplingType == "sequential")

    CPChiddenGar, CPChiddenEncoder = args.hiddenGar, args.hiddenEncoder

    batchSize = args.nGPU * args.batchSizeGPU
    cpcModel.supervised = args.supervised

    if isinstance(cpcModel, model.CPCModelNullspace) or isinstance(cpcModel, model.CPCModelPCA):
        downsampling = cpcModel.cpc.gEncoder.DOWNSAMPLING 
    elif isinstance(cpcModel, model.MultiLevelModel):
        downsampling = cpcModel.frameLevelModel.gEncoder.DOWNSAMPLING
    else:
        downsampling = cpcModel.gEncoder.DOWNSAMPLING
    # Training criterion
    if args.load is not None and args.loadCriterion and not args.freezeFrameModel:
        cpcCriterion = loadCriterion(args.load[0],  downsampling,
                                     len(speakers), nPhones)
    else:
        cpcCriterion = getCriterion(args, downsampling,
                                    len(speakers), nPhones)

    if loadOptimizer and not args.freezeFrameModel:
        state_dict = torch.load(args.load[0], 'cpu')
        cpcCriterion.load_state_dict(state_dict["cpcCriterion"])

    cpcCriterion.cuda()
    cpcModel.cuda()
    
    auxCriterion = None
    predictClusterIdCriterion = None
    # Optimizer
    if args.freezeFrameModel:
        print("Working with frozen features at frame level")
        cpcModel.frameLevelModel.eval()        
        for g in cpcModel.frameLevelModel.parameters():
            g.requires_grad = False

    if args.codebookLearningRate is not None:
        gParams = [
            {
                'params': utils.getParametersForOptimizer(cpcModel, onlyCodebook=True),
                'lr': args.codebookLearningRate
            },
            {
                'params': utils.getParametersForOptimizer(cpcModel, withCodebook=False),
                'lr': args.learningRate
            },
            {
                'params': utils.getParametersForOptimizer(cpcCriterion, onlyCodebook=True),
                'lr': args.codebookLearningRate
            },
            {
                'params': utils.getParametersForOptimizer(cpcCriterion, withCodebook=False),
                'lr': args.learningRate
            }
        ]
    else:
        gParams = list(cpcCriterion.parameters()) + list(cpcModel.parameters())
    if args.useKreukLoss:
        auxCriterion = sa.CPCUnsupersivedCriterion(1,
                                                    1,
                                                    args.hiddenEncoder,
                                                    args.hiddenEncoder,
                                                    1,
                                                    allowed_skips_beg=args.CPCCTCSkipBeg,
                                                    allowed_skips_end=args.CPCCTCSkipEnd,
                                                    predict_self_loop=args.CPCCTCSelfLoop,
                                                    learn_blank=args.CPCCTCLearnBlank,
                                                    normalize_enc=args.CPCCTCNormalizeEncs,
                                                    normalize_preds=args.CPCCTCNormalizePreds,
                                                    masq_rules=args.CPCCTCMasq,
                                                    loss_temp=args.CPCCTCLossTemp,
                                                    no_negs_in_match_window=args.CPCCTCNoNegsMatchWin,
                                                    limit_negs_in_batch=args.limitNegsInBatch,
                                                    mode=args.cpc_mode,
                                                    rnnMode='none',
                                                    dropout=False,
                                                    speakerEmbedding=args.speakerEmbedding,
                                                    nSpeakers=len(speakers),
                                                    sizeInputSeq=(args.sizeWindow // downsampling),
                                                    normalizeScore=args.normalizeCPCScore)
        gParams += list(auxCriterion.parameters())
    if args.predictClusterIds:
        predictClusterIdCriterion = cr.PhoneCriterion(args.hiddenEncoder, nPhones, onEncoder=False, linear=True, useLSTM=False)
        gParams += list(predictClusterIdCriterion.parameters())
    
    optimizer = torch.optim.Adam(gParams, lr=args.learningRate,
                                 betas=(args.beta1, args.beta2),
                                 eps=args.epsilon)

    if loadOptimizer and not args.onlyCapture and not args.only_classif_metric and not args.freezeFrameModel:
        print("Loading optimizer " + args.load[0])
        state_dict = torch.load(args.load[0], 'cpu')
        if "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])

    # Checkpoint
    if args.pathCheckpoint is not None and not args.onlyCapture and not args.only_classif_metric:
        if not os.path.isdir(args.pathCheckpoint):
            os.mkdir(args.pathCheckpoint)
        args.pathCheckpoint = os.path.join(args.pathCheckpoint, "checkpoint")
        with open(args.pathCheckpoint + "_args.json", 'w') as file:
            json.dump(vars(args), file, indent=2)

    scheduler = None
    if args.schedulerStep > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    args.schedulerStep,
                                                    gamma=0.5)
    if args.schedulerRamp is not None:
        n_epoch = args.schedulerRamp
        print(f"Ramp activated. n_e = {n_epoch}")
        scheduler_ramp = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                           lr_lambda=lambda epoch: utils.ramp_scheduling_function(
                                                               n_epoch, epoch),
                                                           last_epoch=-1)
        if scheduler is None:
            scheduler = scheduler_ramp
        else:
            scheduler = utils.SchedulerCombiner([scheduler_ramp, scheduler],
                                                [0, args.schedulerRamp])
    if scheduler is not None:
        print(f'Redoing {len(logs["epoch"])} scheduler steps')
        for i in range(len(logs["epoch"])):
            scheduler.step()

    print("cpcModel", cpcModel)
    print("cpcCriterion", cpcCriterion)

    cpcModel = torch.nn.DataParallel(cpcModel,
                                     device_ids=range(args.nGPU)).cuda()
    cpcCriterion = torch.nn.DataParallel(cpcCriterion,
                                         device_ids=range(args.nGPU)).cuda()
    if auxCriterion is not None:
        auxCriterion = torch.nn.DataParallel(auxCriterion,
                                         device_ids=range(args.nGPU)).cuda()
    if predictClusterIdCriterion is not None:
        predictClusterIdCriterion = torch.nn.DataParallel(predictClusterIdCriterion,
                                         device_ids=range(args.nGPU)).cuda()
    
    if args.supervised_classif_metric:

        linsep_batch_size = args.linsepBatchSizeGPU * args.nGPU

        dim_features = CPChiddenEncoder if args.phone_get_encoded else CPChiddenGar
        dim_ctx_features = CPChiddenGar  # for speakers using CNN encodings is not supported; could add but not very useful perhaps

        phoneLabelsData = None
        if args.path_phone_data:
            phoneLabelsData, nPhonesInData = parseSeqLabels(args.path_phone_data)
            
            if not args.CTCphones:
                print(f"Running phone separability with aligned phones")
            else:
                print(f"Running phone separability with CTC loss")

            def constructPhoneCriterionAndOptimizer():
                if not args.CTCphones:
                    # print(f"Running phone separability with aligned phones")
                    phone_criterion = cr.PhoneCriterion(dim_features,
                                                nPhonesInData, args.phone_get_encoded,
                                                nLayers=args.linsep_net_layers)
                else:
                    # print(f"Running phone separability with CTC loss")
                    phone_criterion = cr.CTCPhoneCriterion(dim_features,
                                                    nPhonesInData, args.phone_get_encoded,
                                                    nLayers=args.linsep_net_layers)
                phone_criterion.cuda()
                phone_criterion = torch.nn.DataParallel(phone_criterion, device_ids=range(args.nGPU))

                # Optimizer
                phone_g_params = list(phone_criterion.parameters())

                phone_optimizer = torch.optim.Adam(phone_g_params, lr=args.linsep_lr,
                                            betas=(args.linsep_beta1, args.linsep_beta2),
                                            eps=args.linsep_epsilon)
                
                return phone_criterion, phone_optimizer
        
        if args.speaker_sep:
            print(f"Running speaker separability")

            def constructSpeakerCriterionAndOptimizer():
                speaker_criterion = cr.SpeakerCriterion(dim_ctx_features, len(speakers),
                                                        nLayers=args.linsep_net_layers)
                speaker_criterion.cuda()
                speaker_criterion = torch.nn.DataParallel(speaker_criterion, device_ids=range(args.nGPU))

                speaker_g_params = list(speaker_criterion.parameters())

                speaker_optimizer = torch.optim.Adam(speaker_g_params, lr=args.linsep_lr,
                                            betas=(args.linsep_beta1, args.linsep_beta2),
                                            eps=args.linsep_epsilon)

                return speaker_criterion, speaker_optimizer

        linsep_db_train = AudioBatchData(args.pathDB, args.sizeWindow, seqTrain,
                                phoneLabelsData, len(speakers))
        linsep_db_val = AudioBatchData(args.pathDB, args.sizeWindow, seqVal,
                                    phoneLabelsData, len(speakers))

        linsep_train_loader = linsep_db_train.getDataLoader(linsep_batch_size, "uniform", True,
                                        numWorkers=0)

        linsep_val_loader = linsep_db_val.getDataLoader(linsep_batch_size, 'sequential', False,
                                    numWorkers=0)

        def runLinsepClassificationTraining(numOfEpoch, cpcMdl, cpcStateEpoch):
            log_path_for_epoch = os.path.join(args.linsep_logs_dir, str(numOfEpoch))
            if not os.path.exists(log_path_for_epoch):
                os.makedirs(log_path_for_epoch)
            log_path_phoneme = os.path.join(log_path_for_epoch, "phoneme/")
            log_path_speaker = os.path.join(log_path_for_epoch, "speaker/")
            if not os.path.exists(log_path_phoneme):
                os.makedirs(log_path_phoneme)
            if not os.path.exists(log_path_speaker):
                os.makedirs(log_path_speaker)
            if args.linsep_checkpoint_dir:
                checpoint_path_for_epoch = os.path.join(args.linsep_checkpoint_dir, str(numOfEpoch))
                checkpoint_path_phoneme = os.path.join(checpoint_path_for_epoch, "phoneme/")
                checkpoint_path_speaker = os.path.join(checpoint_path_for_epoch, "speaker/")
                if not os.path.exists(checkpoint_path_phoneme):
                    os.makedirs(checkpoint_path_phoneme)
                if not os.path.exists(checkpoint_path_speaker):
                    os.makedirs(checkpoint_path_speaker)
            locLogsPhone = {}
            locLogsSpeaker = {}
            if args.path_phone_data:
                phone_criterion, phone_optimizer = constructPhoneCriterionAndOptimizer()
                locLogsPhone = linsep.trainLinsepClassification(
                    cpcMdl,
                    phone_criterion,  # combined with classification model before
                    linsep_train_loader,
                    linsep_val_loader,
                    phone_optimizer,
                    log_path_phoneme,
                    args.linsep_task_logging_step,
                    checkpoint_path_phoneme,
                    args.linsep_n_epoch,
                    cpcStateEpoch,
                    'phone')
                del phone_criterion
                del phone_optimizer
            if args.speaker_sep:
                speaker_criterion, speaker_optimizer = constructSpeakerCriterionAndOptimizer()
                locLogsSpeaker = linsep.trainLinsepClassification(
                    cpcMdl,
                    speaker_criterion,  # combined with classification model before
                    linsep_train_loader,
                    linsep_val_loader,
                    speaker_optimizer,
                    log_path_speaker,
                    args.linsep_task_logging_step,
                    checkpoint_path_speaker,
                    args.linsep_n_epoch,
                    cpcStateEpoch,
                    'speaker')
                del speaker_criterion
                del speaker_optimizer

            locLogsPhone = {"phone_" + k: v for k, v in locLogsPhone.items()}
            locLogsSpeaker = {"speaker_" + k: v for k, v in locLogsSpeaker.items()}
            return {**locLogsPhone, **locLogsSpeaker}

        linsepClassificationTaskConfig = (args.linsep_classif_each_epochs,
                                            runLinsepClassificationTraining)

    else:
        linsepClassificationTaskConfig = (None, None)

    if not args.onlyCapture and not args.only_classif_metric:
        run(trainDataset,
            valDataset,
            (captureDataset, captureOptions, captureSetStatsCollector),
            linsepClassificationTaskConfig,
            batchSize,
            args.samplingType,
            cpcModel,
            cpcCriterion,
            args.nEpoch,
            args.pathCheckpoint,
            optimizer,
            scheduler,
            logs,
            args.headWeights,
            args.path_phone_data, auxCriterion, predictClusterIdCriterion)
    if args.onlyCapture:  
    # caution [!] - will capture for last checkpoint (last saved state) if checkpoint directory given
    #               to use specific checkpoint provide full checkpoint file path
    #               will use "last state" and not "best in internal CPC accuracy" anyway
        onlyCapture(
            (captureDataset, captureOptions, captureSetStatsCollector),
            batchSize,
            cpcModel,
            cpcCriterion,
            logs)
    if args.only_classif_metric:
    # caution [!] - will use last checkpoint (last saved state) if checkpoint directory given
    #               to use specific checkpoint provide full checkpoint file path
    #               will use "last state" and not "best in internal CPC accuracy" anyway
        trainedEpoch = len(logs["epoch"]) - 1
        # runPhonemeClassificationTraining created above if args.supervised_classif_metric
        runLinsepClassificationTraining(trainedEpoch, cpcModel, trainedEpoch)


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')

    print(len(argv))

    # Default arguments:
    parser = set_default_cpc_config(parser)

    group_db = parser.add_argument_group('Dataset')
    # group_db.add_argument('--pathDB', type=str, default=None,
    #                       help='Path to the directory containing the '
    #                       'data.')
    group_db.add_argument('--pathDB', type=str, nargs="+", default=None,
                          help='Path(s) to the directory containing the '
                          'data.')
    group_db.add_argument('--file_extension', type=str, nargs="+", default=[".flac"],
                          help="Extension(s) of the audio files in the dataset(s).")
    group_db.add_argument('--pathTrain', type=str, nargs="+", default=None,
                          help='Path(s) to a .txt file containing the list of the '
                          'training sequences.')
    group_db.add_argument('--pathVal', type=str, nargs="+", default=None,
                          help='Path(s) to a .txt file containing the list of the '
                          'validation sequences.')
    # stuff below for capturing data
    group_db.add_argument('--onlyCapture', action='store_true',
                          help='Only capture data from learned model for one epoch, ignore training; '
                          'conflicts with pathTrain, pathVal etc. arguments')
    group_db.add_argument('--pathCaptureDS', type=str, default=None,
                          help='Path to a .txt file containing the list of the '
                          'data capturing sequences; additionally it can be specified to log a total number of N, or n percent of set '
                          '(e.g. pass validation path and specify to sample from that)')
    group_db.add_argument('--captureDSfreq', type=int, default=None,
                          help='percentage of pathCaptureDS set to use for capturing; conflicts with --captureDStotNr')
    group_db.add_argument('--captureDStotNr', type=int, default=None,
                          help='total number of *AUDIO FILES* to capture data for; number of chunks will be different.')
    # end of capturing data part here
    group_db.add_argument('--n_process_loader', type=int, default=8,
                          help='Number of processes to call to load the '
                          'dataset')
    group_db.add_argument('--ignore_cache', action='store_true',
                          help='Activate if the dataset has been modified '
                          'since the last training session.')
    group_db.add_argument('--max_size_loaded', type=int, default=4000000000,
                          help='Maximal amount of data (in byte) a dataset '
                          'can hold in memory at any given time')
    group_db.add_argument('--gru_level', type=int, default=-1,
                          help='Hidden level of the LSTM autoregressive model to be taken'
                          '(default: -1, last layer).')

    group_supervised = parser.add_argument_group(
        'Supervised mode (depreciated)')
    group_supervised.add_argument('--supervised', action='store_true',
                                  help='(Depreciated) Disable the CPC loss and activate '
                                  'the supervised mode. By default, the supervised '
                                  'training method is the speaker classification.')
    # group_supervised.add_argument('--pathPhone', type=str, default=None,
    #                               help='(Supervised mode only) Path to a .txt '
    #                               'containing the phone labels of the dataset. If given '
    #                               'and --supervised, will train the model using a '
    #                               'phone classification task.')
    group_supervised.add_argument('--CTC', action='store_true')

    group_supervised_data = parser.add_argument_group(
        'Group with args for passing supervised data both for additional metric-producing classification task, '
        'and for data capturing')
    group_supervised_data.add_argument('--path_phone_data', type=str, default=None,
                        help="Path to the phone labels. If given, with --supervised_classif_metric will be able "
                        'to learn phone classification, with capturing will be able to capture phone alignments')

    group_supervised_metric = parser.add_argument_group(
        'Mode with computing additional supervised phoneme classification accuracy, withou influencing CPC training')
    group_supervised_metric.add_argument('--supervised_classif_metric',
                        action='store_true', help='Compute the metric')
    group_supervised_metric.add_argument('--speaker_sep', action='store_true',
                        help="If given, will"
                        " compute the speaker separability.")
    group_supervised_metric.add_argument('--CTCphones', action='store_true',
                        help="Use the CTC loss (for phone separability only)")
    group_supervised_metric.add_argument('--linsepBatchSizeGPU', type=int, default=8,
                        help='Batch size per GPU for phoneme classification.')
    group_supervised_metric.add_argument('--linsep_n_epoch', type=int, default=10)
    group_supervised_metric.add_argument('--phone_get_encoded', action='store_true',
                        help="If activated, will work with the output of the "
                        " convolutional encoder (see CPC's architecture).")
    group_supervised_metric.add_argument('--linsep_lr', type=float, default=2e-4,
                        help='Learning rate for phoneme classification.')
    group_supervised_metric.add_argument('--linsep_beta1', type=float, default=0.9,
                        help='Value of beta1 for the Adam optimizer for phoneme classification.')
    group_supervised_metric.add_argument('--linsep_beta2', type=float, default=0.999,
                        help='Value of beta2 for the Adam optimizer for phoneme classification.')
    group_supervised_metric.add_argument('--linsep_epsilon', type=float, default=2e-8,
                        help='Value of epsilon for the Adam optimizer for phoneme classification.')
    group_supervised_metric.add_argument('--only_classif_metric',
                        action="store_true", 
                        help="Don't train CPC, just compute classification accuracy on given checkpoint "
                        '(classification net itself is trained) and store in given path; '
                        'conflicts with regular CPC training; need to specify --supervised_classif_metric '
                        'and corresponding args')
    group_supervised_metric.add_argument('--linsep_logs_dir', type=str, default=None,
                        help='Path (root) where to log more detailed phoneme classification training data.')
    group_supervised_metric.add_argument('--linsep_checkpoint_dir', type=str, default=None,
                        help='Path (root) where to save best checkpoint for each classification training performed.')
    group_supervised_metric.add_argument('--linsep_task_logging_step', type=int, default=1,
                        help='how often to save detailed phoneme classification training data')
    group_supervised_metric.add_argument('--linsep_classif_each_epochs', type=int, default=20,
                        help='How often to perform classification task - classification net is then '
                        'trained on train DS representations and assesed on val DS representations '
                        'that are produced after that epoch in eval mode')
    group_supervised_metric.add_argument('--linsep_net_layers', type=int, default='1',
                        help='Description of how big net to use for classification (layers have num_phonemes neurons) ' 
                        'with 1, there is just a linear net used without additional hidden layers')
    
    group_stats = parser.add_argument_group(
        'Args for specifying stats to compute for validation and capture DS')
    # group_stats.add_argument('--valSetStats', type=str, default=None,
    #                     help='For validation DS.')
    # validation DS has smaller number of info - will need to specify stats accordingly
    group_stats.add_argument('--captureSetStats', type=str, default=None,
                        help='For capture DS.')

    group_save = parser.add_argument_group('Save')
    group_save.add_argument('--pathCheckpoint', type=str, default=None,
                            help="Path of the output directory.")
    group_save.add_argument('--logging_step', type=int, default=1000)
    group_save.add_argument('--save_step', type=int, default=5,
                            help="Frequency (in epochs) at which a checkpoint "
                            "should be saved")

    # stuff below for capturing data
    group_save.add_argument('--pathCaptureSave', type=str, default=None, )
    group_save.add_argument('--captureEachEpochs', type=int, default=10, help='how often to save capture data')
    group_save.add_argument('--captureConvRepr', action='store_true', help='if to save representations after the encoder')
    group_save.add_argument('--captureCtxRepr', action='store_true', help='if to save LSTM-based contexts produced in CPC model')
    group_save.add_argument('--captureSpeakerAlign', action='store_true', help='if to save speaker alignments')
    group_save.add_argument('--capturePhoneAlign', action='store_true', help='if to save phone alignments')
    group_save.add_argument('--captureEverything', action='store_true', help='save everything valid in this config')
    # below ONLY for CPC-CTC
    group_save.add_argument('--capturePred', action='store_true', help='if to save CPC predictions')
    group_save.add_argument('--captureCPCCTCalign', action='store_true', help='if to save CTC alignments with CPC predictions - only for CPC-CTC variant')
    group_save.add_argument('--captureCPCCTClogScores', action='store_true', help='if to save alignment log scores')
    # end of capturing data part here

    group_load = parser.add_argument_group('Load')
    group_load.add_argument('--load', type=str, default=None, nargs='*',
                            help="Load an exsiting checkpoint. Should give a path "
                            "to a .pt file. The directory containing the file to "
                            "load should also have a 'checkpoint.logs' and a "
                            "'checkpoint.args'")
    group_load.add_argument('--loadCriterion', action='store_true',
                            help="If --load is activated, load the state of the "
                            "training criterion as well as the state of the "
                            "feature network (encoder + AR)")
    group_load.add_argument('--restart', action='store_true',
                            help="If any checkpoint is found, ignore it and "
                            "restart the training from scratch.")
    group_load.add_argument('--restartEpochCount', action='store_true',
                            help="If any checkpoint is found, ignore it and "
                            "restart the training from scratch.")
    group_load.add_argument('--nullspace', action='store_true',
                            help="Additionally load nullspace")

    group_gpu = parser.add_argument_group('GPUs')
    group_gpu.add_argument('--nGPU', type=int, default=-1,
                           help="Number of GPU to use (default: use all "
                           "available GPUs)")
    group_gpu.add_argument('--batchSizeGPU', type=int, default=8,
                           help='Number of batches per GPU.')
    parser.add_argument('--debug', action='store_true',
                        help="Load only a very small amount of files for "
                        "debugging purposes.")
    parser.add_argument('--useKreukLoss', action='store_true',
                        help="Add a Kreuk-like loss to try to improve segmentations.")
    parser.add_argument('--predictClusterIds', action='store_true',
                        help="Add a loss for pseudo label prediction.")    
    args = parser.parse_args(argv)

    if args.pathDB is None and (args.pathCheckpoint is None or args.restart):
        parser.print_help()
        print("Either provides an input dataset or a checkpoint to load")
        sys.exit()

    if args.pathCheckpoint is not None:
        args.pathCheckpoint = os.path.abspath(args.pathCheckpoint)

    if args.load is not None:
        args.load = [os.path.abspath(x) for x in args.load]

    # set it up if needed, so that it is dumped along with other args
    if args.random_seed is None:
        args.random_seed = random.randint(0, 2**31)

    if args.nGPU < 0:
        args.nGPU = torch.cuda.device_count()
    assert args.nGPU <= torch.cuda.device_count(),\
        f"number of GPU asked: {args.nGPU}," \
        f"number GPU detected: {torch.cuda.device_count()}"
    print(f"Let's use {args.nGPU} GPUs!")

    if args.arMode == 'no_ar':
        args.hiddenGar = args.hiddenEncoder
    return args


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
