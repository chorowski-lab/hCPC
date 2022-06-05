import torch
from torch import autograd, nn
import torch.nn.functional as F
from zmq import device

from .criterion import BaseCriterion, EqualizedConv1d, FFNetwork, ShiftedConv
from ..model import RobustKMeansQuantizer
from ..utils.misc import Globals

class Identity(nn.Module):

    def __init__(self, *args):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class _SOFT_ALIGN(autograd.Function):
    """Soft-align a set of predictions to some vectors.

    Args:
    - log_probs: BS x Num_X x Num_Preds giving log(X|P) (that is the probaility of emission and not symbol classification)

    Retursn:
    - costs (BS), alignments (BS x Num_X) if int's denoting which symbol best fits the value.

    """
    @staticmethod
    def _alignment_cost(log_probs, allowed_skips_beg, allowed_skips_end, force_forbid_blank):
        # log_probs is BS x WIN_LEN x NUM_PREDS
        bs, win_len, num_preds = log_probs.size()
        assert win_len >=  num_preds
        padded_log_probs = F.pad(
            log_probs, (0, 0, allowed_skips_beg, allowed_skips_end), "constant", 0)
        padded_win_len = win_len + allowed_skips_beg + allowed_skips_end
        fake_ctc_labels = torch.arange(1, num_preds+1, dtype=torch.int).expand(bs, num_preds)

        # append impossible BLANK probabilities
        ctc_log_probs = padded_log_probs.permute(1, 0, 2).contiguous()
        if force_forbid_blank:
            ctc_log_probs = torch.cat((
                torch.empty(padded_win_len, bs, 1, device=log_probs.device).fill_(-1000),
                ctc_log_probs
            ), 2)
        # Now ctc_log_probs is win_size x BS x (num_preds + 1)
        assert ctc_log_probs.is_contiguous()

        # normalize the log-probs over num_preds
        # This is required, because ctc returns a bad gradient when given 
        # unnormalized log probs
        log_sum_exps = torch.logsumexp(ctc_log_probs, 2, keepdim=True)
        ctc_log_probs = ctc_log_probs - log_sum_exps
        losses = F.ctc_loss(
            ctc_log_probs, 
            fake_ctc_labels,
            torch.empty(bs, dtype=torch.int).fill_(padded_win_len),
            torch.empty(bs, dtype=torch.int).fill_(num_preds),
            reduction='none')
        losses = losses - log_sum_exps.squeeze(2).sum(0)
        return losses

    @staticmethod
    def forward(ctx, log_probs, allowed_skips_beg=0, allowed_skips_end=0, force_forbid_blank=True):
        log_probs = log_probs.detach().requires_grad_()
        with torch.enable_grad():
            losses = _SOFT_ALIGN._alignment_cost(
                log_probs, allowed_skips_beg, allowed_skips_end, force_forbid_blank)
            losses.sum().backward()
            grads = log_probs.grad.detach()
        _, alignment = grads.min(-1)
        ctx.save_for_backward(grads)

        return losses.detach(), alignment

    @staticmethod
    def backward(ctx, grad_output, _):
        grads, = ctx.saved_tensors
        grad_output = grad_output.to(grads.device)
        return grads * grad_output.view(-1, 1, 1), None, None, None

soft_align = _SOFT_ALIGN.apply


class PredictionNetwork(nn.Module):

    def __init__(self,
                 nPredicts,
                 dimOutputAR,
                 dimOutputEncoder,
                 rnnMode=None,
                 dropout=False,
                 sizeInputSeq=116):

        super(PredictionNetwork, self).__init__()
        self.predictors = nn.ModuleList()
        self.RESIDUAL_STD = 0.01
        self.dimOutputAR = dimOutputAR

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

    def forward(self, c):

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
            locC = locC.view(locC.size(0), locC.size(1), locC.size(2), 1)
            out.append(locC)
        return torch.cat(out, 3)




class CPCUnsupersivedCriterion(BaseCriterion):

    def __init__(self,
                 nPredicts,             # Number of predictions
                 nMatched,                  # Window size to which align predictions
                 dimOutputAR,           # Dimension of G_ar
                 dimOutputEncoder,      # Dimension of the convolutional net
                 negativeSamplingExt,   # Number of negative samples to draw
                 allowed_skips_beg=0,     # number of predictions that we can skip at the beginning
                 allowed_skips_end=0,     # number of predictions that we can skip at the end
                 predict_self_loop=False, # always predict a repetition of the first symbol
                 no_negs_in_match_window=False,  # prevent sampling negatives from the matching window
                 learn_blank=False,       # try to use the blank symbol
                 normalize_enc=False,
                 normalize_preds=False,
                 masq_rules="",
                 loss_temp=1.0,
                 limit_negs_in_batch=None,
                 mode=None,
                 rnnMode=None,
                 dropout=False,
                 speakerEmbedding=0,
                 nSpeakers=0,
                 sizeInputSeq=128,
                 normalizeScore=False,
                 targetQuantizer=None,
                 adjacentNegatives=False,
                 rlSetup='vanillaReinforce'):

        super(CPCUnsupersivedCriterion, self).__init__()
        if speakerEmbedding > 0:
            print(
                f"Using {speakerEmbedding} speaker embeddings for {nSpeakers} speakers")
            self.speakerEmb = torch.nn.Embedding(nSpeakers, speakerEmbedding)
            dimOutputAR += speakerEmbedding
        else:
            self.speakerEmb = None

        self.rnnMode = rnnMode
        self.normalize_enc = normalize_enc
        self.normalize_preds = normalize_preds
        self.loss_temp = loss_temp
        self.nPredicts = nPredicts
        self.adjacentNegatives = adjacentNegatives
        self.nMatched = nPredicts if self.adjacentNegatives else nMatched
        self.negativeSamplingExt = 1 if self.adjacentNegatives else negativeSamplingExt
        self.margin = self.nMatched + 1 if self.adjacentNegatives else self.nMatched
        self.no_negs_in_match_window = no_negs_in_match_window
        self.wPrediction = PredictionNetwork(
            nPredicts, dimOutputAR, dimOutputEncoder, rnnMode=rnnMode,
            dropout=dropout, sizeInputSeq=sizeInputSeq - self.margin)
        self.learn_blank = learn_blank
        if learn_blank:
            self.blank_proto = torch.nn.Parameter(torch.zeros(1, 1, dimOutputEncoder, 1))
        else:
            self.register_parameter('blank_proto', None)        
        self.allowed_skips_beg = allowed_skips_beg
        self.allowed_skips_end = allowed_skips_end
        self.predict_self_loop = predict_self_loop
        # if predict_self_loop:
        #     self.self_loop_gain = torch.nn.Parameter(torch.ones(1))
        # else:
        #     self.register_parameter('self_loop_gain', None)
        self.limit_negs_in_batch = limit_negs_in_batch

        if masq_rules:
            masq_buffer = torch.zeros(self.nMatched, self.nPredicts)
            for rule in masq_rules.split(','):
                a,b,c,d = [int(a) if a.lower() != "none" else None for a in rule.split(':')]
                masq_buffer[a:b,c:d] = 1
            print("!!!MasqBuffer: ", masq_buffer)
            self.register_buffer("masq_buffer", masq_buffer.unsqueeze(0))
        else:
            self.register_buffer("masq_buffer", None)

        if mode not in [None, "reverse"]:
            raise ValueError("Invalid mode")

        self.mode = mode
        self.normalizeScore = normalizeScore
        self.maxSizeInputSeq = sizeInputSeq
        self.targetQuantizer = targetQuantizer
        self.numUpdates = 0
        self.register_buffer('baseline', torch.Tensor([0]))
        # self.baseline = None
        self.rlSetup = rlSetup
        self.EMA_COEFF = 0.99

    def computeExtraLosses(self, device):
        loss = {}
        if self.targetQuantizer is not None and not isinstance(self.targetQuantizer, RobustKMeansQuantizer):
            loss['targetQuantizerLoss'] = self.targetQuantizer.computeLoss().view(1, 1)
        return loss
    
    def updateCounter(self):
        self.numUpdates += 1
        if self.targetQuantizer is not None:
            self.targetQuantizer.numUpdates += 1

    def sampleClean(self, targets, windowSizes, maxWindowSize):

        batchSize, maxPooledLen, dimEncoded = targets.size()

        if not self.adjacentNegatives:
            batchIdx = torch.randint(low=0, high=batchSize,
                                    size=(batchSize, maxWindowSize * self.negativeSamplingExt, ),
                                    device=targets.device)
            if self.limit_negs_in_batch:
                # sample negatives from a small set of entries in minibatch
                batchIdx = torch.remainder(batchIdx, self.limit_negs_in_batch)
                batchBaseIdx = torch.arange(0, batchSize, device=targets.device)
                batchBaseIdx -= torch.remainder(batchBaseIdx, self.limit_negs_in_batch)
                batchIdx += batchBaseIdx.unsqueeze(1)
                # we can get too large, if batchsize is not divisible by limit_negs_in_batch
                batchIdx = torch.remainder(batchIdx, batchSize)
        else: # We sample within a neighborhood, therefore we sample from the same
            batchIdx = torch.arange(0, batchSize, device=targets.device).view(batchSize, 1).repeat(
                1, maxWindowSize * self.nMatched)

        batchIdx = batchIdx.contiguous().view(-1, 1)
        compressedLens = windowSizes + self.nMatched
        seqLens = compressedLens[batchIdx].view(-1, 1).to(targets.device)

        if not self.adjacentNegatives:
            if self.no_negs_in_match_window:
                idx_low = self.nMatched  # forbid sampling negatives in the prediction window
            else:
                idx_low = 1  # just forbid sampling own index for negative

            seqIdx = torch.randint(low=idx_low, high=maxPooledLen, 
                                size=(batchSize, maxWindowSize, self.negativeSamplingExt),
                                device=targets.device)
            seqBaseIdx = torch.arange(0, maxWindowSize, device=targets.device).unsqueeze(0).unsqueeze(2)
            seqIdx += seqBaseIdx
            seqIdx = seqIdx.view(batchSize * maxWindowSize* self.negativeSamplingExt, 1)
            seqIdx = torch.remainder(seqIdx, seqLens)
        else:
            # -------------------------------------- NEW --------------------------------------
            seqBaseIdx = torch.arange(0, maxWindowSize, device=targets.device).view(1, -1, 1).repeat(
                batchSize, 1, self.nMatched)
            predOffset = torch.arange(1, self.nMatched + 1, device=targets.device).view(1, 1, -1)
            negOffset = torch.LongTensor([-1, 1]).to(targets.device)[torch.randint(0, 2, 
                (batchSize * maxWindowSize * self.nMatched,), device=targets.device)].view(batchSize, maxWindowSize, self.nMatched)
            # negOffset[:, :, 0] = 1
            seqIdx = (seqBaseIdx + predOffset + negOffset).view(-1, 1)
            # -------------------------------------- NEW --------------------------------------

            # -------------------------------------- OLD --------------------------------------
            # # assert (self.negativeSamplingExt + 2) <= self.nMatched # Just to ensure we won't be sampling from the padded regions
            # seqBaseIdx = torch.arange(0, maxWindowSize, device=targets.device).view(1, -1, 1).repeat(
            #     batchSize, 1, self.negativeSamplingExt)
            # negIdxs = torch.arange(2, self.negativeSamplingExt + 2, device=targets.device).view(1, 1, -1)
            # seqIdx = (seqBaseIdx + negIdxs).view(-1, 1)
            # # seqIdx[seqIdx >= seqLens] = (seqBaseIdx - negIdxs).view(-1, 1)[seqIdx >= seqLens]
            # -------------------------------------- OLD --------------------------------------
        sampledNegs = targets[batchIdx, seqIdx, :].contiguous().view(-1, dimEncoded).view(batchSize, maxWindowSize, -1, dimEncoded)
        
        return sampledNegs

        
    def forward(self, cFeature, encodedData, label, captureOptions=None, computePGLoss=True):
        batchSize = label.size(0)
        boundaryLogProbs = None

        if isinstance(cFeature, dict):
            seqSize = cFeature['segmentSeqLens']
            boundaryLogProbs = cFeature['boundaryLogProbs']
            # segmentLens = cFeature['segmentLens']
            cFeature = cFeature['paddedCFeatures']
        else:
            seqSize = torch.IntTensor([cFeature.size(1)] * batchSize)
        
        if self.mode == "reverse":
            encodedData = torch.flip(encodedData, [1])
            cFeature = torch.flip(cFeature, [1])
        
        seqSize = torch.maximum(seqSize, torch.ones_like(seqSize) * (self.margin + 1))
        windowSizes = seqSize - self.margin

        if self.normalize_enc:
            encodedData = F.layer_norm(encodedData, (encodedData.size(-1),))

        # tooShortBatchIdx = torch.where(windowSizes < 52)[0]
        # if len(tooShortBatchIdx) > 0:
        #     for i in tooShortBatchIdx:
        #         numReps = 52 - windowSizes[i]
        #         fixedC = cFeature[i]
        #         repeats = torch.ones((fixedC.size(0),))
        #         fixedC = fixedC
        #         cFeature[i, seqSize[i]:seqSize[i] + numReps, :] = cFeature[i, seqSize[i] - 1, :]
        #         encodedData[i, seqSize[i]:seqSize[i] + numReps, :] = encodedData[i, seqSize[i] - 1, :]

        # windowSizes = torch.maximum(windowSizes, torch.ones_like(windowSizes))
        maxWindowSize = torch.max(windowSizes)
        cFeature = cFeature[:, :maxWindowSize]
        targets = encodedData[:, :maxWindowSize + self.margin]
        
        if self.targetQuantizer is not None:
            targets = self.targetQuantizer(targets, seqSize) # .detach()

        # negatives: BS x Len x NumNegs x D if not adjacentNegatives else BS x Len x nPred x D
        sampledNegs = self.sampleClean(targets, windowSizes, maxWindowSize)

        if self.speakerEmb is not None:
            l_ = label.view(batchSize, 1).expand(batchSize, maxWindowSize)
            embeddedSpeaker = self.speakerEmb(l_)
            cFeature = torch.cat([cFeature, embeddedSpeaker], dim=2)

        if self.rnnMode == 'transformer':
            cFeature = F.pad(cFeature, (0, 0, 0, self.maxSizeInputSeq - self.margin - maxWindowSize))
        # Predictions, BS x Len x D x nPreds
        predictions = self.wPrediction(cFeature)[:, :maxWindowSize, :, :]
        nPredicts = self.nPredicts

        extra_preds = []

        if self.learn_blank:
            extra_preds.append(self.blank_proto.expand(batchSize, maxWindowSize, self.blank_proto.size(2), 1))

        if self.predict_self_loop:
            # old and buggy
            # extra_preds.append(cFeature.unsqueeze(-1))
            # new and shiny
            extra_preds.append(encodedData[:, :maxWindowSize, :].unsqueeze(-1) )  # * self.self_loop_gain)

        if extra_preds:
            nPredicts += len(extra_preds)
            extra_preds.append(predictions)
            predictions = torch.cat(
                extra_preds, -1
            )

        if self.normalize_preds:
            predictions = F.layer_norm(predictions, (predictions.size(-1),))
        
        # Positive examples in the window, BS x Len x W x D
        marginPositives = self.margin - self.nMatched
        positives = targets[:, 1:(-marginPositives) if marginPositives > 0 else None].unfold(1, self.nMatched, 1).permute(0, 1, 3, 2)
        # gt_and_neg = torch.cat((pred_windows, sampledData.permute(0, 2, 3, 1)), 3)

        # BS x L x NumNegs x NumPreds
        # BS x L x NumNegs x NumPreds if not adjacentNegatives else BS x L x NumPreds x D
        if self.adjacentNegatives:
            # Predictions, BS x Len x D x nPreds
            # negatives: BS x Len x NumNegs x D if not adjacentNegatives else BS x Len x nPreds x D
            # neg_log_scores: BS x Len x 1 x nPreds
            neg_log_scores = (sampledNegs.transpose(2, 3) * predictions).sum(2, keepdim=True)
        else:
            neg_log_scores = sampledNegs @ predictions
        # BS x L x W x NumPreds
        pos_log_scores = positives @ predictions
        if self.normalizeScore:
            EPS = 1e-12
            normNegs = torch.sqrt((sampledNegs * sampledNegs).sum(-1, keepdim=True)) # BS x Len x nPreds x 1
            if self.adjacentNegatives:
                normNegs = normNegs.transpose(2, 3)
            normPos = torch.sqrt((positives * positives).sum(-1, keepdim=True))
            normPreds = torch.sqrt((predictions * predictions).sum(-2, keepdim=True))
            neg_log_scores /= (normNegs + EPS)
            neg_log_scores /= (normPreds + EPS)
            pos_log_scores /= (normPos + EPS)
            pos_log_scores /= (normPreds + EPS)
        else:
            neg_log_scores /= sampledNegs.size(-1)
            pos_log_scores /= sampledNegs.size(-1)

        # We now want ot get a matrix BS x L x W x NumPreds
        # in which each entry is the log-softmax of predicting a window elem in contrast to al negs

        # log(e^x_p / (e^x_p + \sum_n e^x_n))
        # first compute \log \sum_n e^x_n
        neg_log_tot_scores = torch.logsumexp(neg_log_scores, 2, keepdim=True)

        # now log(e^xp / (e^x_p + e^x_n)) 
        # this can be further optimized.
        paddedLogScores = torch.log_softmax(torch.stack((pos_log_scores, neg_log_tot_scores.expand_as(pos_log_scores)), 0), dim=0)[0]
        paddingSize = paddedLogScores.size(1) - windowSizes
        splitSizes = torch.cat([windowSizes.view(-1, 1), paddingSize.view(-1, 1)], dim=1).view(-1)
        log_scores = torch.cat(torch.split(paddedLogScores.view(-1, paddedLogScores.size(2), paddedLogScores.size(3)), tuple(splitSizes))[::2])
        # log_scores = torch.nn.utils.rnn.pack_padded_sequence(paddedLogScores, windowSizes.cpu(), batch_first=True, enforce_sorted=False)
        # log_scores = log_scores.data

        # print('ls-stats', log_scores.mean().item(), log_scores.std().item())
        if self.masq_buffer is not None:
            masq_buffer = self.masq_buffer
            if extra_preds:
                masq_buffer = torch.cat([masq_buffer[:, :, :1]] * (len(extra_preds) - 1) + [masq_buffer], dim=2)
            log_scores = log_scores.masked_fill(masq_buffer > 0, -1000)
        seqLosses, aligns = soft_align(log_scores / self.loss_temp, self.allowed_skips_beg, self.allowed_skips_end, not self.learn_blank)
        seqLosses = seqLosses * self.loss_temp
        pos_is_selected = (pos_log_scores > neg_log_scores.max(2, keepdim=True)[0]).view(batchSize*maxWindowSize, self.nMatched, nPredicts)
        
        # This is approximate Viterbi alignment loss and accurracy
        outLosses = -torch.gather(log_scores, 2, aligns.unsqueeze(-1)).view(-1, self.nMatched, 1)
        
        if boundaryLogProbs is not None and computePGLoss:
            # We will compute a PG loss as well
            boundaryLogProbs = boundaryLogProbs[:, :maxWindowSize] # Just consider the segments for which we have computed the CPC loss
            boundaryLogProbs = torch.cat(torch.split(boundaryLogProbs.contiguous().view(-1), tuple(splitSizes))[::2])
            # segmentLens = segmentLens[:, :maxWindowSize]
            # segmentLens = torch.cat(torch.split(segmentLens.contiguous().view(-1), tuple(splitSizes))[::2])
            
            # reward = seqLosses.detach()
            reward = outLosses.mean(-2).view(-1).detach()
            # # The reward at any position is the average of the CPC loss at that position and the two previous ones
            # paddedSeqLosses = torch.nn.utils.rnn.pad_sequence(
            #     torch.split(seqLosses, tuple(windowSizes)), batch_first=True
            #     )
            # paddedSeqLosses = F.pad(paddedSeqLosses, (2, 0))
            # reward = F.unfold(paddedSeqLosses.unsqueeze(1).unsqueeze(3), (2, 1), dilation=2).sum(1)
            # if reward.size(1) > 1:
            #     reward[:, 1:] /= 2
            #     # reward[:, 1] /= 2
            #     # if reward.size(1) > 2:
            #     #     reward[:, 2:] /= 3
            # reward = torch.cat(torch.split(reward.view(-1), tuple(splitSizes))[::2])

            paddedLogProbs = torch.nn.utils.rnn.pad_sequence(torch.split(boundaryLogProbs, tuple(windowSizes)), batch_first=True)
            paddedRewards = torch.nn.utils.rnn.pad_sequence(torch.split(reward, tuple(windowSizes)), batch_first=True)

            if self.training:
                if self.baseline.item() == 0:
                    self.baseline += reward.mean()
                else:
                    # torch.abs(self.baseline - (self.baseline * self.EMA_COEFF + (1 - self.EMA_COEFF) * reward.mean())) > 0,3
                    self.baseline *= self.EMA_COEFF
                    self.baseline += ((1 - self.EMA_COEFF) * reward.mean())
                Globals.writer.add_scalar(tag='REINFORCE baseline', scalar_value=self.baseline.item(), global_step=Globals.currentIteration)

            if self.rlSetup == 'vanillaReinforce':                    
                policyGradLoss = (paddedLogProbs.sum(1) * paddedRewards.mean(1)).mean()
            elif self.rlSetup == 'reinforceWBaseline':                    
                policyGradLoss = (paddedLogProbs.sum(1) * (paddedRewards.mean(1) - self.baseline)).mean()
            elif self.rlSetup == 'reinforceWAdvantage':
                advantage = reward - self.baseline
                paddedAdvantage = torch.nn.utils.rnn.pad_sequence(torch.split(advantage, tuple(windowSizes)), batch_first=True)
                policyGradLoss = (paddedLogProbs * paddedAdvantage).sum(1).mean()
            # policyGradLoss = policyGradLoss * 0.1
        
        outLosses = outLosses.squeeze(-1).float().mean(0, keepdim=True)
        outAcc = torch.gather(pos_is_selected, 2, aligns.unsqueeze(-1)).view(-1, self.nMatched, 1)
        outAcc = outAcc.squeeze(-1).float().mean(0, keepdim=True)
        # just simulate a per-prediction loss
        outLossesD = outLosses.detach()
        losses = seqLosses.mean() / (outLossesD.sum() + 1e-12) * outLossesD

        captureRes = None
        if captureOptions != None:
            for o in captureOptions:
                assert o in ('pred', 'cpcctc_align', 'cpcctc_log_scores', 'locals', 'seqLosses', 'emaSeqLosses')
            captureRes = {}
            if 'pred' in captureOptions:
                # 1st sting in last dim can be self loop - need to keep as it's also being aligned
                captureRes['pred'] = predictions
            if 'cpcctc_align' in captureOptions:
                readableAligns = aligns.detach().view(batchSize, maxWindowSize, self.nMatched)
                captureRes['cpcctc_align'] = readableAligns
            if 'cpcctc_log_scores' in captureOptions:
                captureRes['cpcctc_log_scores'] = log_scores.detach().view(batchSize, maxWindowSize, self.nMatched, -1)
            if 'locals' in captureOptions:
                captureRes['locals'] = locals()
            if 'seqLosses' in captureOptions:
                captureRes['seqLosses'] = seqLosses
            if 'emaSeqLosses' in captureOptions:
                captureRes['emaSeqLosses'] = self.baseline
        extraLosses = self.computeExtraLosses(label.device)
        if boundaryLogProbs is not None and computePGLoss:
            extraLosses['piGradLoss'] = policyGradLoss.view(1, 1)
        return [losses], [outAcc], [captureRes], extraLosses


class MultiLevelCriterion(BaseCriterion):    
    def __init__(self,
                 frameLevelCriterion,
                 nPredicts,             # Number of predictions
                 nMatched,              # Window size to which align predictions
                 dimOutputAR,           # Dimension of S_ar
                 dimOutputEncoder,      # Dimension of S_enc
                 negativeSamplingExt,   # Number of negative samples to draw
                 sizeInputSeq=128,
                 normalizeScore=False,
                 targetQuantizer=None,
                 adjacentNegatives=False,
                 rlSetup='vanillaReinforce'):
        super(MultiLevelCriterion, self).__init__()

        self.frameLevelCriterion = frameLevelCriterion
        self.segmentLevelCriterion = CPCUnsupersivedCriterion(
                nPredicts, 
                nMatched, 
                dimOutputAR, 
                dimOutputEncoder,
                negativeSamplingExt,
                sizeInputSeq=sizeInputSeq,
                normalizeScore=normalizeScore,
                rnnMode='transformer',
                targetQuantizer=targetQuantizer,
                adjacentNegatives=adjacentNegatives,
                rlSetup=rlSetup
        )
        self.numUpdates = 0

    def updateCounter(self):
        self.numUpdates += 1
        if self.frameLevelCriterion is not None:
            self.frameLevelCriterion.updateCounter()
        self.segmentLevelCriterion.updateCounter()
        
    def forward(self, cFeature, encodedData, label, captureOptions=None, computePGLoss=True):
        losses = []
        outAcc = []
        capRes = []
        extraLosses = {}
        if self.frameLevelCriterion is not None:
            lossesFrameLvl, accuraciesFrameLevel, captureResFrameLevel, extraLosses = self.frameLevelCriterion(cFeature[0], encodedData[0], label, captureOptions)
            losses += lossesFrameLvl
            outAcc += accuraciesFrameLevel
            capRes += captureResFrameLevel
        lossesSegmentLvl, accuraciesSegmentLevel, captureResSegmentLevel, extraLossesSeg = self.segmentLevelCriterion(cFeature[1], encodedData[1], label, captureOptions, computePGLoss)
        extraLosses.update(extraLossesSeg)
        losses += lossesSegmentLvl
        outAcc += accuraciesSegmentLevel
        capRes += captureResSegmentLevel
        # if cFeature[0].device.index == 1 and self.segmentLevelCriterion.targetQuantizer is not None:
        #     self.segmentLevelCriterion.targetQuantizer.reestimationReservoir.buffer = torch.zeros(
        #         self.segmentLevelCriterion.targetQuantizer.reestimationReservoir.n, 
        #         encodedData[1].size(-1), 
        #         device=encodedData[1].device
        #         )
        return losses, outAcc, capRes, extraLosses