# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_layers import EqualizedLinear, EqualizedConv1d
from cpc.criterion.seq_alignment import collapseLabelChain, getSeqPER
import random
import math

class Identity(nn.Module):

    def __init__(self, *args):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class FFNetwork(nn.Module):
    def __init__(self, din, dout, dff, dropout):
        super(FFNetwork, self).__init__()
        self.lin1 = EqualizedLinear(din, dff, bias=True, equalized=True)
        self.lin2 = EqualizedLinear(dff, dout, bias=True, equalized=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.drop(self.relu(self.lin1(x))))


class ShiftedConv(nn.Module):
    def __init__(self, dimOutputAR, dimOutputEncoder, kernelSize):
        super(ShiftedConv, self).__init__()
        self.module = EqualizedConv1d(dimOutputAR, dimOutputEncoder,
                                      kernelSize, equalized=True,
                                      padding=0)
        self.kernelSize = kernelSize

    def forward(self, x):

        # Input format: N, S, C -> need to move to N, C, S
        N, S, C = x.size()
        x = x.permute(0, 2, 1)

        padding = torch.zeros(N, C, self.kernelSize - 1, device=x.device)
        x = torch.cat([padding, x], dim=2)
        x = self.module(x)
        x = x.permute(0, 2, 1)
        return x


class PredictionNetwork(nn.Module):

    def __init__(self,
                 nPredicts,
                 dimOutputAR,
                 dimOutputEncoder,
                 simMeasure,
                 rnnMode=None,
                 dropout=False,
                 sizeInputSeq=116):

        super(PredictionNetwork, self).__init__()
        self.predictors = nn.ModuleList()
        self.RESIDUAL_STD = 0.01
        self.dimOutputAR = dimOutputAR
        self.simMeasure = simMeasure

        self.dropout = nn.Dropout(p=0.5) if dropout else None
        for i in range(nPredicts):
            if rnnMode == 'RNN':
                self.predictors.append(
                    nn.RNN(dimOutputAR, dimOutputEncoder))
                self.predictors[-1].flatten_parameters()
            elif rnnMode == 'LSTM':
                self.predictors.append(
                    nn.LSTM(dimOutputAR, dimOutputEncoder, batch_first=True))
                self.predictors[-1].flatten_parameters()
            elif rnnMode == 'ffd':
                self.predictors.append(
                    FFNetwork(dimOutputAR, dimOutputEncoder,
                              dimOutputEncoder, 0))
            elif rnnMode == 'conv4':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 4))
            elif rnnMode == 'conv8':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 8))
            elif rnnMode == 'conv12':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 12))
            elif rnnMode == 'transformer':
                from cpc.transformers import buildTransformer
                self.predictors.append(
                    buildTransformer(dimOutputEncoder,
                                       1,
                                       sizeInputSeq,
                                       False, True))
            elif rnnMode == 'none':
                self.predictors.append(Identity())
            else:
                self.predictors.append(
                    nn.Linear(dimOutputAR, dimOutputEncoder, bias=False))
                if dimOutputEncoder > dimOutputAR:
                    residual = dimOutputEncoder - dimOutputAR
                    self.predictors[-1].weight.data.copy_(torch.cat([torch.randn(
                        dimOutputAR, dimOutputAR), self.RESIDUAL_STD * torch.randn(residual, dimOutputAR)], dim=0))

    def forward(self, c, candidates):

        assert(len(candidates) == len(self.predictors))
        out = []

        # UGLY
        if isinstance(self.predictors[0], EqualizedConv1d):
            c = c.permute(0, 2, 1)

        for k in range(len(self.predictors)):

            locC = self.predictors[k](c)
            if isinstance(locC, tuple):
                locC = locC[0]
            if isinstance(self.predictors[k], EqualizedConv1d):
                locC = locC.permute(0, 2, 1)
            if self.dropout is not None:
                locC = self.dropout(locC)
            locC = locC.view(locC.size(0), 1, locC.size(1), locC.size(2))
            if self.simMeasure == 'dotproduct':
                outK = (locC*candidates[k]).mean(dim=3)
            elif self.simMeasure == 'cosine':
                outK = F.cosine_similarity(locC, candidates[k], dim=3)
            else:
                raise NotImplementedError            
            out.append(outK)       
        return out


class BaseCriterion(nn.Module):

    def warmUp(self):
        return False

    def update(self):
        return


class NoneCriterion(BaseCriterion):
    def __init__(self):
        super(NoneCriterion, self).__init__()

    def forward(self, cFeature, encodedData, label):
        return torch.zeros(1, 1, device=cFeature.device), \
            torch.zeros(1, 1, device=cFeature.device)


class CPCUnsupersivedCriterion(BaseCriterion):

    def __init__(self,
                 nPredicts,             # Number of steps
                 dimOutputAR,           # Dimension of G_ar
                 dimOutputEncoder,      # Dimension of the convolutional net
                 negativeSamplingExt,   # Number of negative samples to draw
                 normalizeScore=False,
                 mode=None,
                 rnnMode=False,
                 dropout=False,
                 speakerEmbedding=0,
                 nSpeakers=0,
                 sizeInputSeq=128):

        super(CPCUnsupersivedCriterion, self).__init__()
        if speakerEmbedding > 0:
            print(
                f"Using {speakerEmbedding} speaker embeddings for {nSpeakers} speakers")
            self.speakerEmb = torch.nn.Embedding(nSpeakers, speakerEmbedding)
            dimOutputAR += speakerEmbedding
        else:
            self.speakerEmb = None

        self.wPrediction = PredictionNetwork(
            nPredicts, dimOutputAR, dimOutputEncoder, simMeasure='cosine' if normalizeScore else 'dotproduct', rnnMode=rnnMode,
            dropout=dropout, sizeInputSeq=sizeInputSeq - nPredicts)
        self.nPredicts = nPredicts
        self.negativeSamplingExt = negativeSamplingExt
        self.lossCriterion = nn.CrossEntropyLoss()

        if mode not in [None, "reverse"]:
            raise ValueError("Invalid mode")

        self.mode = mode

    def sampleClean(self, encodedData, windowSize):

        batchSize, nNegativeExt, dimEncoded = encodedData.size()
        outputs = []

        negExt = encodedData.contiguous().view(-1, dimEncoded)
        # Draw nNegativeExt * batchSize negative samples anywhere in the batch
        batchIdx = torch.randint(low=0, high=batchSize,
                                 size=(self.negativeSamplingExt
                                       * windowSize * batchSize, ),
                                 device=encodedData.device)

        seqIdx = torch.randint(low=1, high=nNegativeExt,
                               size=(self.negativeSamplingExt
                                     * windowSize * batchSize, ),
                               device=encodedData.device)

        baseIdx = torch.arange(0, windowSize, device=encodedData.device)
        baseIdx = baseIdx.view(1, 1,
                               windowSize).expand(1,
                                                  self.negativeSamplingExt,
                                                  windowSize).expand(batchSize, self.negativeSamplingExt, windowSize)
        seqIdx += baseIdx.contiguous().view(-1)
        seqIdx = torch.remainder(seqIdx, nNegativeExt)

        extIdx = seqIdx + batchIdx * nNegativeExt
        negExt = negExt[extIdx].view(batchSize, self.negativeSamplingExt,
                                     windowSize, dimEncoded)

        labelLoss = torch.zeros((batchSize * windowSize),
                                dtype=torch.long,
                                device=encodedData.device)

        for k in range(1, self.nPredicts + 1):

            # Positive samples
            if k < self.nPredicts:
                posSeq = encodedData[:, k:-(self.nPredicts-k)]
            else:
                posSeq = encodedData[:, k:]

            posSeq = posSeq.view(batchSize, 1, posSeq.size(1), dimEncoded)
            fullSeq = torch.cat((posSeq, negExt), dim=1)
            outputs.append(fullSeq)

        return outputs, labelLoss

    def getInnerLoss(self):

        return "orthoLoss", self.orthoLoss * self.wPrediction.orthoCriterion()

    def forward(self, cFeature, encodedData, label, captureOptions=None):
        if self.mode == "reverse":
            encodedData = torch.flip(encodedData, [1])
            cFeature = torch.flip(cFeature, [1])

        batchSize, seqSize, dimAR = cFeature.size()
        windowSize = seqSize - self.nPredicts

        cFeature = cFeature[:, :windowSize]

        sampledData, labelLoss = self.sampleClean(encodedData, windowSize)

        if self.speakerEmb is not None:
            l_ = label.view(batchSize, 1).expand(batchSize, windowSize)
            embeddedSpeaker = self.speakerEmb(l_)
            cFeature = torch.cat([cFeature, embeddedSpeaker], dim=2)

        predictions = self.wPrediction(cFeature, sampledData)

        captureRes = None
        if captureOptions != None:
            for o in captureOptions:
                assert o in ('pred',)
            captureRes = {}
            if 'pred' in captureOptions:
                assert False   # not supported yet, predictions here are in some very weird format it seems
                captureRes['pred'] = None

        outLosses = [0 for x in range(self.nPredicts)]
        outAcc = [0 for x in range(self.nPredicts)]

        for k, locPreds in enumerate(predictions[:self.nPredicts]):
            locPreds = locPreds.permute(0, 2, 1)
            locPreds = locPreds.contiguous().view(-1, locPreds.size(2))
            lossK = self.lossCriterion(locPreds, labelLoss)
            outLosses[k] += lossK.view(1, -1)
            _, predsIndex = locPreds.max(1)
            outAcc[k] += torch.sum(predsIndex == labelLoss).float().view(1, -1)

        return [torch.cat(outLosses, dim=1)], \
            [torch.cat(outAcc, dim=1) / (windowSize * batchSize)], \
                captureRes


class SpeakerCriterion(BaseCriterion):

    def __init__(self, dimEncoder, nSpeakers, nLayers=1):

        super(SpeakerCriterion, self).__init__()
        # self.linearSpeakerClassifier = nn.Linear(
        #     dimEncoder, nSpeakers)
        if nLayers == 1:
            self.linearSpeakerClassifier = nn.Linear(dimEncoder, nSpeakers)
        else:
            outLayers = [nn.Linear(dimEncoder, nSpeakers)]
            for l in range(nLayers - 1):
                outLayers.append(nn.ReLU())
                outLayers.append(nn.Linear(nSpeakers, nSpeakers))
            self.linearSpeakerClassifier = nn.Sequential(*outLayers)
        self.lossCriterion = nn.CrossEntropyLoss()
        self.entropyCriterion = nn.LogSoftmax(dim=1)

    def forward(self, cFeature, otherEncoded, label, computeAccuracy=None):

        # cFeature.size() : batchSize x seq Size x hidden size
        batchSize = cFeature.size(0)
        cFeature = cFeature[:, -1, :]
        cFeature = cFeature.view(batchSize, -1)
        predictions = self.linearSpeakerClassifier(cFeature)

        loss = self.lossCriterion(predictions, label).view(1, -1)
        acc = (predictions.max(1)[1] == label).double().mean().view(1, -1)

        return loss, acc

class SpeakerDoubleCriterion(BaseCriterion):

    def __init__(self, dimEncoder, dimInter, nSpeakers):

        super(SpeakerDoubleCriterion, self).__init__()
        self.linearSpeakerClassifier = nn.Sequential(nn.Linear(dimEncoder, dimInter), 
            nn.Linear(dimInter, nSpeakers))
        self.lossCriterion = nn.CrossEntropyLoss()
        self.entropyCriterion = nn.LogSoftmax(dim=1)

    def forward(self, cFeature, otherEncoded, label, computeAccuracy=None):

        # cFeature.size() : batchSize x seq Size x hidden size
        batchSize = cFeature.size(0)
        cFeature = cFeature[:, -1, :]
        cFeature = cFeature.view(batchSize, -1)
        predictions = self.linearSpeakerClassifier(cFeature)

        loss = self.lossCriterion(predictions, label).view(1, -1)
        acc = (predictions.max(1)[1] == label).double().mean().view(1, -1)

        return loss, acc


class PhoneCriterion(BaseCriterion):

    def __init__(self, dimEncoder, nPhones, onEncoder, linear=False, useLSTM=True, 
                 useConvClassifier=False):

        super(PhoneCriterion, self).__init__()
        self.useLSTM = useLSTM
        self.useConvClassifier = useConvClassifier
        d = 2 if useLSTM else 1
        if linear:
            self.PhoneCriterionClassifier = nn.Linear(dimEncoder * d, nPhones)
        elif useConvClassifier:
            self.PhoneCriterionClassifier = torch.nn.Conv1d(
                dimEncoder * d, nPhones, 8, stride=4)
        else:
            self.PhoneCriterionClassifier = nn.Sequential(
                nn.Linear(dimEncoder * d, dimEncoder * 2*d),
                nn.ReLU(),
                nn.Linear(dimEncoder * 2*d, nPhones),
            )
        if useLSTM:
            self.lstm = torch.nn.LSTM(dimEncoder, dimEncoder, num_layers=1, batch_first=True, bidirectional=True)

        self.lossCriterion = nn.CrossEntropyLoss()
        self.onEncoder = onEncoder

    def forward(self, cFeature, otherEncoded, label, computeAccuracy=None):

        # cFeature.size() : batchSize x seq Size x hidden size
        if self.onEncoder:
            predictions = self.getPrediction(otherEncoded)
        else:
            predictions = self.getPrediction(cFeature)
        predictions = predictions.view(-1, predictions.size(2))
        label = label.view(-1)
        loss = self.lossCriterion(predictions, label).view(1, -1)
        acc = (predictions.max(1)[1] == label).double().mean().view(1, -1)
        return loss, acc

    def getPrediction(self, x):
        if self.useLSTM:
            try:
                self.lstm.flatten_parameters()
            except RuntimeError:
                pass
            x = self.lstm(x)[0]
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)[0]
        if self.useConvClassifier:
            x = x.permute(0, 2, 1)
            x = self.PhoneCriterionClassifier(x)
            return x.permute(0, 2, 1)
        else:
            return self.PhoneCriterionClassifier(x)


class CTCPhoneCriterion(BaseCriterion):

    def __init__(self, dimEncoder, nPhones, onEncoder, linear=False, useLSTM=True, 
                 useConvClassifier=False,forbid_blank=False, upsample=False):

        super(CTCPhoneCriterion, self).__init__()
        self.useLSTM = useLSTM
        self.useConvClassifier = useConvClassifier
        self.upsample = upsample
        d = 2 if useLSTM else 1
        if linear:
            self.PhoneCriterionClassifier = nn.Linear(dimEncoder * d, nPhones + 1)
        elif useConvClassifier:
            self.PhoneCriterionClassifier = torch.nn.Conv1d(
                dimEncoder * d, nPhones + 1, 8, stride=4)
        else:
            self.PhoneCriterionClassifier = nn.Sequential(
                nn.Linear(dimEncoder * d, dimEncoder * 2*d),
                nn.ReLU(),
                nn.Linear(dimEncoder * 2*d, nPhones + 1),
            )
        # if upsample:
            # self.upsampleLayer = torch.nn.ConvTranspose1d(nPhones + 1, nPhones + 1, kernel_size=1, 
                                                        #   stride=3)
        if useLSTM:
            self.lstm = torch.nn.LSTM(dimEncoder, dimEncoder, num_layers=1, batch_first=True, bidirectional=True)
        self.lossCriterion = nn.CTCLoss(blank=nPhones, zero_infinity=True)
        self.onEncoder = onEncoder
        self.BLANK_LABEL = nPhones
        self.forbid_blank = forbid_blank

    def extra_repr(self):
        return f"CTCPhoneCriterion(..., onEncoder={self.onEncoder}, forbid_blank={self.forbid_blank})"

    def getPrediction(self, x):
        if self.useLSTM:
            try:
                self.lstm.flatten_parameters()
            except RuntimeError:
                pass
            x = self.lstm(x)[0]
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)[0]
        if self.useConvClassifier:
            x = x.permute(0, 2, 1)
            x = self.PhoneCriterionClassifier(x)
            return x.permute(0, 2, 1)
        else:
            return self.PhoneCriterionClassifier(x)

    def forward(self, cFeature, otherEncoded, label, computeAccuracy=False):
        onDownsampledHead = isinstance(cFeature, dict)
        if onDownsampledHead:
            targetSizePred = cFeature['seqLens']
            otherEncoded = cFeature['encodedData']
            cFeature = cFeature['states']
            B, _, H = cFeature.size()
            if self.useLSTM:
                cFeature = torch.nn.utils.rnn.pack_padded_sequence(cFeature, targetSizePred.cpu(), batch_first=True, enforce_sorted=False)
                otherEncoded = torch.nn.utils.rnn.pack_padded_sequence(otherEncoded, targetSizePred.cpu(), batch_first=True, enforce_sorted=False)
        else:
            B, S, H = cFeature.size()
            targetSizePred = torch.ones(B, dtype=torch.int64,
                                    device=cFeature.device) * S
        if self.useConvClassifier:
            targetSizePred = ((targetSizePred - 8) // 4 + 1)

        features = otherEncoded if self.onEncoder else cFeature
        predictions = self.getPrediction(features)
        if self.forbid_blank:
            predictions += (
                -1e4 * 
                (torch.arange(self.BLANK_LABEL+1, device=predictions.device) == self.BLANK_LABEL
                ).float().view(1, 1, self.BLANK_LABEL+1))
        predictions = torch.nn.functional.log_softmax(predictions, dim=2)
        label = label.to(predictions.device)
        label, sizeLabels = collapseLabelChain(label)
        if self.upsample:
            tooShortBatchIdx = torch.where(targetSizePred < sizeLabels)[0]
            if len(tooShortBatchIdx) > 0:
                expandBy = torch.ceil(torch.max(sizeLabels[tooShortBatchIdx] / targetSizePred[tooShortBatchIdx])).int()
                predictions = predictions.repeat_interleave(expandBy, dim=1)
                targetSizePred *= expandBy
        loss = self.lossCriterion(predictions.permute(1, 0, 2), label,
                                  targetSizePred, sizeLabels).view(1, -1)
        avgPER = 0.
        if computeAccuracy:
            for b in range(B):
                predictedPhones = predictions[b].max(1)[1].detach().cpu()
                predictedPhone, sizePredictions = collapseLabelChain(torch.unsqueeze(predictedPhones[:targetSizePred[b]], dim=0))
                predictedPhone = predictedPhone[predictedPhone != self.BLANK_LABEL]
                avgPER += getSeqPER((predictedPhone, label[b, :sizeLabels[b]].cpu()))
            avgPER /= B
        return loss, avgPER * torch.ones(1, 1, device=loss.device)


class ModelCriterionCombined(torch.nn.Module):
    def __init__(self, model, criterion):
        super(ModelCriterionCombined, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, data, label):
        c_feature, encoded_data, label = self.model(data, label)
        loss, acc = self.criterion(c_feature, encoded_data, label)
        return loss, acc
