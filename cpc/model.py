# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import random
# import re
# from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import time

import torch
from cpc.utils.misc import compressBatch, getCompressionMatrix, Globals, get_mask1d
from sklearn import cluster
import numpy as np
from .utils.misc import Globals

###########################################
# Networks
###########################################

class Jitter(nn.Module):
    def __init__(self, prob=0.12, **kwargs):
        super(Jitter, self).__init__()
        self.prob = prob

    def forward(self, x):
        if self.training:
            _, l, _ = x.size()
            index = torch.arange(0, l).to(x)
            change = torch.bernoulli(index * 0 + self.prob * 2) # whether to change
            shift = torch.bernoulli(index * 0 + 0.5) * 2 - 1 # left or right
            index = index + change * shift
            index = index.long().clamp(0, l - 1)
            x = torch.index_select(x, dim=1, index=index)
        return x


class IDModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super(IDModule, self).__init__()

    def forward(self, x):
        return x


class ChannelNorm(nn.Module):

    def __init__(self,
                 numFeatures,
                 epsilon=1e-05,
                 affine=True):

        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1,
                                                              numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(2)

        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean)*torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class SincConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, 
                 padding_mode='zeros', sampleRate=16000, minLowHz=50, minBandHz=50):
        super(SincConv1D, self).__init__()
        if in_channels != 1:
            msg = "SincConv1D only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        self.outChannels = out_channels
        self.kernelSize = kernel_size
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if self.kernelSize % 2 == 0:
            self.kernelSize += 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if bias:
            raise ValueError('SincConv1D does not support bias.')
        if groups > 1:
            raise ValueError('SincConv1D does not support groups.')
        self.sampleRate = sampleRate
        self.minLowHz = minLowHz
        self.minBandHz = minBandHz
        # Initialize filterbanks such that they are equally spaced in Mel scale
        lowHz = 30
        highHz = self.sampleRate / 2 - (self.minLowHz + self.minBandHz)
        mel = np.linspace(self.hz2Mel(lowHz), self.hz2Mel(highHz), self.outChannels + 1)
        hz = self.mel2Hz(mel)
        # Filter lower frequency (outChannels, 1)
        self.lowHz_ = torch.nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        # Filter frequency band (outChannels, 1)
        self.bandHz_ = torch.nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        # Hamming window
        nLin= torch.linspace(0, (self.kernelSize / 2) - 1, 
                             steps=int((self.kernelSize / 2))) # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * nLin / self.kernelSize)
        n = (self.kernelSize - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sampleRate # Due to symmetry, we only need half of the time axes
    
    @staticmethod
    def hz2Mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def mel2Hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, waveforms):
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.minLowHz  + torch.abs(self.lowHz_)
        high = torch.clamp(low + self.minBandHz + torch.abs(self.bandHz_), self.minLowHz, self.sampleRate/2)
        band = (high - low)[:, 0]
        fTimesTLow = torch.matmul(low, self.n_)
        fTimesTHigh = torch.matmul(high, self.n_)
        # Equivalent of Eq.4 of the reference paper
        bandPassLeft = ((torch.sin(fTimesTHigh) - torch.sin(fTimesTLow)) / (self.n_/2)) * self.window_ 
        bandPassCenter = 2 * band.view(-1, 1)
        bandPassRight = torch.flip(bandPassLeft, dims=[1])
        bandPass = torch.cat([bandPassLeft, bandPassCenter, bandPassRight], dim=1)
        bandPass = bandPass / (2 * band[:, None])
        self.filters = (bandPass).view(self.outChannels, 1, self.kernelSize)
        return torch.conv1d(waveforms, self.filters, stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, bias=None, groups=1) 


class CPCEncoder(nn.Module):

    def __init__(self,
                 sizeHidden=512,
                 normMode="layerNorm",
                 linearOutput=False,
                 sincNet=False):

        super(CPCEncoder, self).__init__()

        validModes = ["batchNorm", "instanceNorm", "ID", "layerNorm"]
        if normMode not in validModes:
            raise ValueError(f"Norm mode must be in {validModes}")

        if normMode == "instanceNorm":
            def normLayer(x): return nn.InstanceNorm1d(x, affine=True)
        elif normMode == "ID":
            normLayer = IDModule
        elif normMode == "layerNorm":
            normLayer = ChannelNorm
        else:
            normLayer = nn.BatchNorm1d

        self.dimEncoded = sizeHidden
        if sincNet:
            self.conv0 = SincConv1D(1, sizeHidden, 10, stride=5, padding=3)
        else:
            self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4,
                               stride=2, padding=1)
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = normLayer(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160
        self.linearOutput = linearOutput

    def getDimOutput(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        if self.linearOutput:
            x = self.batchNorm4(self.conv4(x))
        else:
            x = F.relu(self.batchNorm4(self.conv4(x)))
        return x


class MFCCEncoder(nn.Module):

    def __init__(self,
                 dimEncoded):

        super(MFCCEncoder, self).__init__()
        melkwargs = {"n_mels": max(128, dimEncoded), "n_fft": 321}
        self.dimEncoded = dimEncoded
        self.MFCC = torchaudio.transforms.MFCC(n_mfcc=dimEncoded,
                                               melkwargs=melkwargs)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.MFCC(x)
        return x.permute(0, 2, 1)


class LFBEnconder(nn.Module):

    def __init__(self, dimEncoded, normalize=True):

        super(LFBEnconder, self).__init__()
        self.dimEncoded = dimEncoded
        self.conv = nn.Conv1d(1, 2 * dimEncoded,
                              400, stride=1)
        self.register_buffer('han', torch.hann_window(400).view(1, 1, 400))
        self.instancenorm = nn.InstanceNorm1d(dimEncoded, momentum=1) \
            if normalize else None

    def forward(self, x):

        N, C, L = x.size()
        x = self.conv(x)
        x = x.view(N, self.dimEncoded, 2, -1)
        x = x[:, :, 0, :]**2 + x[:, :, 1, :]**2
        x = x.view(N * self.dimEncoded, 1,  -1)
        x = torch.nn.functional.conv1d(x, self.han, bias=None,
                                       stride=160, padding=350)
        x = x.view(N, self.dimEncoded,  -1)
        x = torch.log(1 + torch.abs(x))

        # Normalization
        if self.instancenorm is not None:
            x = self.instancenorm(x)
        return x


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class GumbelQuantizer(nn.Module):
    def __init__(self,
                 dim,
                 codeDim,
                 numGroups=2,
                 numCodes=320,
                 combineGroups=True,
                 temp=(2.0, 0.5, 0.999988804921253),
                 projDepth=2,
                 projFactor=2,
                 activation=nn.GELU()):
        super(GumbelQuantizer, self).__init__()
        assert (
            codeDim % numGroups == 0
        ), f"dim {codeDim} must be divisible by groups {numGroups} for concatenation"
        varDim = codeDim // numGroups
        self.embeddings = nn.Parameter(torch.FloatTensor(
            1, (numGroups if not combineGroups else 1) * numCodes, varDim))
        nn.init.uniform_(self.embeddings)
        self.numGroups = numGroups
        self.numCodes = numCodes
        self.combineGroups = combineGroups
        self.maxTemp, self.minTemp, self.tempDecay = temp
        if projDepth > 1:
            def block(inputDim, outputDim):
                return nn.Sequential(nn.Linear(inputDim, outputDim), activation)
            innerDim = dim * projFactor
            self.codeProbModel = nn.Sequential(
                *[
                    block(dim if i == 0 else innerDim, innerDim)
                    for i in range(projDepth - 1)
                ],
                nn.Linear(innerDim, numGroups * numCodes),
            )
        else:
            self.codeProbModel = nn.Linear(dim, numGroups * numCodes)
            nn.init.normal_(self.codeProbModel.weight, mean=0, std=1)
            nn.init.zeros_(self.codeProbModel.bias)
        self.numUpdates = 0
        self.avgProbs = None

    def computeLoss(self):
        return -0.1 * ((self.numGroups if not self.combineGroups else 1) * self.numCodes - 0.1 * torch.exp(
            -torch.sum(self.avgProbs * torch.log(self.avgProbs + 1e-7), dim=-1)
        ).sum())

    def forward(self, x):
        bs, l, d = x.size()
        x = x.reshape(bs * l, d)
        x = self.codeProbModel(x)
        x = x.view(bs * l * self.numGroups, -1)
    
        self.avgProbs = torch.softmax(
            x.view(bs * l, self.numGroups, -1).float(), dim=-1
        ).mean(dim=0)

        if self.training:
            gumbels = (
            -torch.empty_like(x, memory_format=torch.legacy_contiguous_format).exponential_().log()
            )  # ~Gumbel(0,1)
            tau = max(self.maxTemp * self.tempDecay ** self.numUpdates, self.minTemp)
            gumbels = (x + gumbels) / tau  # ~Gumbel(logits,tau)
            softX = gumbels.softmax(dim=-1)
            _, index = softX.max(-1, keepdim=True)
            hardX = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
            # Straight through.
            x = hardX + softX - softX.detach()
        else:
            _, index = x.max(-1, keepdim=True)
            x = (
                x.new_zeros(*x.shape)
                .scatter_(-1, index.view(-1, 1), 1.0)
                .view(bs * l, self.numGroups, -1)
            )
        x = x.view(bs * l, -1)
        embeddings = self.embeddings
        if self.combineGroups:
            embeddings = embeddings.repeat(1, self.numGroups, 1)
        x = x.unsqueeze(-1) * embeddings
        x = x.view(bs * l, self.numGroups, self.numCodes, -1)
        x = x.sum(-2)
        
        # self.loss = entropyLoss.view(1,)
        return x.view(bs, l, -1)


class ReservoirSampler(nn.Module):
    def __init__(self, num_samples=1024):
        super(ReservoirSampler, self).__init__()
        self.n = num_samples
        self.ttot = 0
        self.register_buffer('buffer', torch.zeros((num_samples, 256)))
        self.register_buffer('i', torch.IntTensor([0]))
        self.reset()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        buffer_key = prefix + 'buffer'
        if buffer_key in state_dict:
            self.buffer = state_dict[buffer_key]
        return super(ReservoirSampler, self
                     )._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def reset(self):
        self.i *= 0
        self.buffer *= 0

    def add(self, samples):
        self.ttot -= time.time()
        samples = samples.detach()
        # if self.i == 0:
            # self.buffer = torch.empty(
                # self.n, samples.size(-1), device=samples.device)
        buffer = self.buffer
        if self.i < self.n:
            slots = self.n - self.i
            add_samples = samples[:slots]
            samples = samples[slots:]
            buffer[self.i: self.i + len(add_samples)] = add_samples
            self.i += len(add_samples)
            if not len(samples):
                print(f"Res size {self.i.item()}")
                self.ttot += time.time()
                return

        for s in samples:
            # warning, includes right end too.
            idx = random.randint(0, self.i.item())
            self.i += 1
            if idx < len(buffer):
                buffer[idx] = s
        self.ttot += time.time()

    def contents(self):
        return self.buffer[:self.i]


class IndicesComputation(object):

    @staticmethod
    def nearest(inputs, codebook, temperature=None):
        with torch.no_grad():
            # inputs: NxD
            # codebook: KxD
            # NxK
            distances_matrix = torch.cdist(inputs, codebook)
            # Nx1
            if temperature is None:
                indices = torch.min(distances_matrix, dim=-1)[1].unsqueeze(1)
            else:
                probs = F.softmax(-distances_matrix / temperature, dim=-1)
                m = torch.distributions.Categorical(probs)
                indices = m.sample()
            return indices


class VectorQuantization(torch.autograd.Function):

    @staticmethod
    def flatten(x):
        code_dim = x.size(-1)
        return x.view(-1, code_dim)

    @staticmethod
    def restore_shapes(codes, indices, target_shape):
        idx_shape = list(target_shape)
        idx_shape[-1] = 1
        return codes.view(*target_shape), indices.view(*idx_shape)

    @staticmethod
    def forward(ctx, inputs, codebook, commitment=0.25,
                criterion='nearest', detachQuantized=False, criterion_kwargs={},
                use_copy_through=False):
        inputs_flat = VectorQuantization.flatten(inputs)
        compute_indices = getattr(IndicesComputation, criterion)
        indices = compute_indices(inputs_flat, codebook, **criterion_kwargs)
        if type(indices) is tuple:
            indices, values = indices
        codes = codebook[indices.view(-1), :]
        codes, indices = VectorQuantization.restore_shapes(
            codes, indices, inputs.shape)
        ctx.save_for_backward(codes, inputs, torch.FloatTensor([commitment]),
                              codebook, indices, torch.tensor([use_copy_through]), torch.tensor([detachQuantized]))
        ctx.mark_non_differentiable(indices)
        return codes, indices

    @staticmethod
    def backward(ctx, straight_through, unused_indices, unused_values=None):
        (codes, inputs, beta, codebook, indices, use_copy_through, detachQuantized) = ctx.saved_tensors
        # print('Magumbos')
        # print(straight_through.size())
        # print(straight_through)
        # assert False

        # TODO: figure out proper vq loss reduction
        vq_loss = F.mse_loss(inputs, codes).detach()
        Globals.writer.add_scalar(tag='VQ loss', scalar_value=vq_loss, global_step=Globals.currentIteration)

        # gradient of vq_loss
        diff = 2 * (inputs - codes) / inputs.numel()

        commitment = beta.item() * diff

        if use_copy_through.item():
            code_disp = VectorQuantization.flatten(-diff + straight_through)
        else:
            code_disp = VectorQuantization.flatten(-diff)
        indices = VectorQuantization.flatten(indices)
        code_disp = (torch
                     .zeros_like(codebook)
                     .index_add_(0, indices.view(-1), code_disp))
        if detachQuantized.item():
            return commitment, code_disp, None, None, None, None    
        return straight_through + commitment, code_disp, None, None, None, None


quantize = VectorQuantization.apply


class RobustKMeansQuantizer(nn.Module):
    def __init__(self,
                 dim,
                 codeDim,
                 numCodes=320,
                 gamma=0.25,
                 reestimationReservoirSize=20480,
                 reestimateEveryEpochs=1,
                 reestimateEveryIters=1.5,
                 reestimateMaxIters=0,
                 reestimateMaxEpochs=50,
                 bottleneckEnforceFromEpoch=3,
                 reestimateEveryItersExpansion=True,
                 logInputNorms=True,
                 logCodeUsage=True
                 ):
        super(RobustKMeansQuantizer, self).__init__()
        self.batchNorm = nn.BatchNorm1d(dim)
        self.embedding = nn.Embedding(numCodes, codeDim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.codeDim = codeDim
        self.commitment = gamma
        self.register_buffer(
            'reestimationData',
            torch.tensor([
                Globals.epoch,
                Globals.currentIteration,
                reestimateEveryEpochs,  # next reestimation epoch
                int(reestimateEveryIters),  # next reestimation iter
                1,  # Reestimation is operating
            ], dtype=torch.int32))
        if reestimationReservoirSize:
            self.reestimationReservoir = ReservoirSampler(reestimationReservoirSize)
            self.reestimateEveryEpochs = reestimateEveryEpochs
            self.reestimateLastEpoch = Globals.epoch
            self.reestimateEveryIters = reestimateEveryIters
            self.reestimateEveryItersExpansion = reestimateEveryItersExpansion
            self.reestimateMaxEpochs = reestimateMaxEpochs
            self.reestimateMaxIters = reestimateMaxIters
            assert reestimateEveryEpochs or reestimateEveryIters
        else:
            self.reestimationReservoir = None
        self.quantizationEnforceFromEpoch = bottleneckEnforceFromEpoch
        self.logInputNorms = logInputNorms
        self.logCodeUsage = logCodeUsage
        self.numUpdates = 0

    def reestimate(self):
        #
        # When warming up, we keep the encodings from the last epoch and
        # reestimate just before new epoch starts.
        # When quantizing, we reestimate every number of epochs or iters given.
        #
        lastEpoch, lastIter, nextReestEpoch, nextReestIter, isOperating = self.reestimationData
        if not isOperating:
            print(f"Re-Disabling reestimation buffer")
            self.reestimationReservoir = None
            return
        if self.quantizationEnforceFromEpoch > 0:
            # Warmup
            if lastEpoch == Globals.epoch:
                return
            # A new epoch has started:
            self.reestimationData[0] = Globals.epoch
            if Globals.epoch < self.quantizationEnforceFromEpoch:
                print("Reseting reservoir")
                self.reestimationReservoir.reset()
                return
            else:
                # We will start quantizing soon, let it run
                pass
        else:
            # Normal operation
            if (self.reestimateEveryEpochs and Globals.epoch < nextReestEpoch):
                return
            if (self.reestimateEveryIters and
                    Globals.currentIteration < nextReestIter):
                return

        # Set the next reestimation iter.
        if self.reestimateEveryItersExpansion:
            nextReestIter = (
                Globals.currentIteration * self.reestimateEveryIters) + 1
        else:
            nextReestIter = (
                Globals.currentIteration + self.reestimateEveryIters)
        self.reestimationData[:4] = torch.tensor([
            Globals.epoch,
            Globals.currentIteration,
            Globals.epoch + self.reestimateEveryEpochs,
            nextReestIter])

        if self.reestimationReservoir.i == 0:
            return
        tstart = time.time()
        numClusters = self.embedding.weight.size(0)
        encodings = self.reestimationReservoir.contents()
        if encodings.size(0) < numClusters:
            print(f"Skipping reestimation, too few samples")
            return
        encodings = encodings.cpu().numpy()
        clustered, *_ = cluster.k_means(encodings, numClusters)
        self.embedding.weight.data[
            ...] = torch.tensor(clustered).to(self.embedding.weight.device)
        self.reestimationReservoir.reset()
        print(f"Done reestimating VQ embedings, took {time.time() - tstart}s")
        if ((self.reestimateMaxEpochs and
             Globals.epoch > self.reestimateMaxEpochs)
            or
            (self.reestimateMaxIters and
             Globals.currentIteration > self.reestimateMaxIters)):
            print(f"Disabling reestimation buffer")
            self.reestimationData[4] = 0
            self.reestimationReservoir = None

    def pack_x(self, x, x_lens):
        if x_lens is None:
            return x
        else:
            mask = get_mask1d(x_lens.to(x.device)).unsqueeze(-1) > 0
            x_sel = torch.masked_select(x, mask)
            x_sel = x_sel.view(mask.sum(), x.size(-1))
            return x_sel

    def unpack_x(self, x, x_lens):
        if x_lens is None:
            return x
        else:
            x_seqs = x.split(tuple(x_lens))
            x = torch.nn.utils.rnn.pad_sequence(
                x_seqs, batch_first=True)
            return x

    def forward(self, x, enc_len=None, returnClusterIds=False):
        if self.reestimationReservoir and self.training and x.device.index == 0:
            self.reestimate()

        if self.logInputNorms and self.training:
            norms = torch.norm(x.contiguous().view(-1, x.size(-1)), dim=1)
            Globals.writer.add_scalar(tag='VQ Input norms pre-BN/mean', scalar_value=torch.mean(norms), global_step=Globals.currentIteration)
            Globals.writer.add_scalar(tag='VQ Input norms pre-BN/std', scalar_value=torch.std(norms), global_step=Globals.currentIteration)

        
        bs, l, d = x.size()
        x = x.transpose(1, 2)
        x = self.batchNorm(x)
        x = x.transpose(1, 2).contiguous()

        if self.logInputNorms and self.training:
            norms = torch.norm(x.view(-1, x.size(-1)), dim=1)
            Globals.writer.add_scalar(tag='VQ Input norms post-BN/mean', scalar_value=torch.mean(norms), global_step=Globals.currentIteration)
            Globals.writer.add_scalar(tag='VQ Input norms post-BN/std', scalar_value=torch.std(norms), global_step=Globals.currentIteration)

        # x = self.projection(x)
        if self.training and self.reestimationReservoir and x.device.index == 0:
            self.reestimationReservoir.add(
                self.pack_x(x, enc_len)
                    .view(-1, x.size(-1)).detach())

        if Globals.epoch < self.quantizationEnforceFromEpoch and not returnClusterIds:
            # print("Skipping quantization")
            codes = x
            indices = torch.zeros(x.shape[:-1] + (1,), device=x.device, dtype=torch.int64)
            values = indices
        else:
            x = self.pack_x(x, enc_len)
            codes, indices = quantize(x, self.embedding.weight, self.commitment, 'nearest', False)
            codes = self.unpack_x(codes, enc_len)
            indices = self.unpack_x(indices, enc_len)

        if self.training:
            self._logCodeUsage(indices, 'nearest')

        if self.logInputNorms and self.training:
            norms = torch.norm(codes.view(-1, codes.size(-1)).contiguous(), dim=1)
            Globals.writer.add_scalar(tag='VQ Input norms post-VQ/mean', scalar_value=torch.mean(norms), global_step=Globals.currentIteration)
            Globals.writer.add_scalar(tag='VQ Input norms post-VQ/std', scalar_value=torch.std(norms), global_step=Globals.currentIteration)

        if returnClusterIds:
            return indices
        return codes

        

    def _logCodeUsage(self, indices, criterion):
        numTokens = self.embedding.weight.size(0)
        codeFreqs = torch.histc(
            indices.float(),
            bins=numTokens, min=-0.5, max=numTokens - 0.5
            ).float()
        count = np.prod(indices.size())
        if criterion != 'sparse':
            assert codeFreqs.sum().item() == count
        codeFreqs /= count
        entropy = torch.distributions.Categorical(codeFreqs).entropy()
        Globals.writer.add_scalar(tag='VQ code usage', scalar_value=entropy.item() / np.log(numTokens), global_step=Globals.currentIteration)


class KMeansQuantizer(nn.Module):
    def __init__(self,
                 dim,
                 codeDim,
                 numGroups=2, 
                 numCodes=320,
                 combineGroups=True,
                 gamma=0.25):
        super(KMeansQuantizer, self).__init__()
        assert (
            codeDim % numGroups == 0
        ), f"dim {codeDim} must be divisible by groups {numGroups} for concatenation"
        self.numGroups = numGroups
        self.numCodes = numCodes
        self.combineGroups = combineGroups
        self.varDim = codeDim // numGroups
        self.embeddings = nn.Parameter(
            0.01 * torch.randn(numCodes, (numGroups if not combineGroups else 1), self.varDim)
        )
        self.projection = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, groups=numGroups, bias=False),
            Fp32GroupNorm(numGroups, dim),
        )
        self.gamma = gamma
        self.mseMean = nn.MSELoss(reduction="mean")
        self.numUpdates = 0
        self.ze = None
        self.zq = None

    @property
    def expandEmbedding(self):
        if self.combineGroups:
            return self.embeddings.expand(self.numCodes, self.numGroups, self.varDim)
        return self.embeddings

    def computeLoss(self):
        latentLoss = self.mseMean(self.zq, self.ze.detach())
        commitmentLoss = self.mseMean(self.ze, self.zq.detach())
        return latentLoss + self.gamma * commitmentLoss

    def forward(self, x, enc_len=None, returnClusterIds=False):
        bs, l, d = x.size()
        x = x.transpose(1, 2)
        ze = self.projection(x)
        ze_ = ze.view(bs, self.numGroups, self.varDim, l).permute(0, 3, 1, 2)
        d = (
            (ze_.unsqueeze(0) - self.expandEmbedding.unsqueeze(1).unsqueeze(1))
            .view(self.numCodes, bs, l, self.numGroups, -1)
            .norm(dim=-1, p=2)
        )
        idx = d.argmin(dim=0)
        if returnClusterIds:
            return idx
        zq = (
            torch.stack(
                [
                    self.expandEmbedding[idx[..., group], group]
                    for group in range(self.numGroups)
                ],
                dim=-2,
            )
            .view(bs, l, self.numGroups * self.varDim)
            .permute(0, 2, 1)
        )
        assert ze.shape == zq.shape, (ze.shape, zq.shape)
        # Straight through.
        x = zq.detach() + ze - ze.detach()
        
        x = x.transpose(1, 2)
        self.ze = ze.float()
        self.zq = zq.float()
        return x


class CPCAR(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 keepHidden,
                 nLevelsGRU,
                 mode="GRU",
                 reverse=False):

        super(CPCAR, self).__init__()
        self.RESIDUAL_STD = 0.1

        if mode == "LSTM":
            self.baseNet = nn.LSTM(dimEncoded, dimOutput,
                                   num_layers=nLevelsGRU, batch_first=True)
        elif mode == "RNN":
            self.baseNet = nn.RNN(dimEncoded, dimOutput,
                                  num_layers=nLevelsGRU, batch_first=True)
        else:
            self.baseNet = nn.GRU(dimEncoded, dimOutput,
                                  num_layers=nLevelsGRU, batch_first=True)

        self.hidden = None
        self.keepHidden = keepHidden
        self.reverse = reverse

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        if self.reverse:
            x = torch.flip(x, [1])
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.baseNet(x, self.hidden)
        if self.keepHidden:
            if isinstance(h, tuple):
                self.hidden = tuple(x.detach() for x in h)
            else:
                self.hidden = h.detach()

        # For better modularity, a sequence's order should be preserved
        # by each module
        if self.reverse:
            x = torch.flip(x, [1])
        return x


class NoAr(nn.Module):

    def __init__(self, numHidden):
        super(NoAr, self).__init__()
        self.dimOutput = numHidden

    def getDimOutput(self):
        return self.dimOutput

    def forward(self, x):
        return x


class BiDIRARTangled(nn.Module):
    r"""
    Research: bidirectionnal model for BERT training.
    """
    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 nLevelsGRU):

        super(BiDIRARTangled, self).__init__()
        assert(dimOutput % 2 == 0)

        self.ARNet = nn.GRU(dimEncoded, dimOutput // 2,
                            num_layers=nLevelsGRU, batch_first=True,
                            bidirectional=True)

    def getDimOutput(self):
        return self.ARNet.hidden_size * 2

    def forward(self, x):

        self.ARNet.flatten_parameters()
        xf, _ = self.ARNet(x)
        return xf


class BiDIRAR(nn.Module):
    r"""
    Research: bidirectionnal model for BERT training.
    """
    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 nLevelsGRU):

        super(BiDIRAR, self).__init__()
        assert(dimOutput % 2 == 0)

        self.netForward = nn.GRU(dimEncoded, dimOutput // 2,
                                 num_layers=nLevelsGRU, batch_first=True)
        self.netBackward = nn.GRU(dimEncoded, dimOutput // 2,
                                  num_layers=nLevelsGRU, batch_first=True)

    def getDimOutput(self):
        return self.netForward.hidden_size * 2

    def forward(self, x):

        self.netForward.flatten_parameters()
        self.netBackward.flatten_parameters()
        xf, _ = self.netForward(x)
        xb, _ = self.netBackward(torch.flip(x, [1]))
        return torch.cat([xf, torch.flip(xb, [1])], dim=2)


###########################################
# Model
###########################################


class CPCModel(nn.Module):

    def __init__(self,
                 encoder,
                 AR,
                 quantizerEncodings=None,
                 quantizerContext=None):

        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR
        self.quantizerEncodings = quantizerEncodings
        self.quantizerContext = quantizerContext
        self.numUpdates = 0

    def computeExtraLosses(self, device):
        loss = {}
        if self.quantizerEncodings is not None:
            loss['quantizerEncodingsLoss'] = self.quantizerEncodings.computeLoss().view(1, 1)
        if self.quantizerContext is not None:
            loss['quantizerContextLoss'] =  self.quantizerContext.computeLoss().view(1, 1)
        return loss

    def updateCounter(self):
        self.numUpdates += 1
        if self.quantizerEncodings is not None:
            self.quantizerEncodings.numUpdates += 1
        if self.quantizerContext is not None:
            self.quantizerContext.numUpdates += 1

    def forward(self, batchData, label):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        if self.quantizerEncodings is not None:
            encodedData = self.quantizerEncodings(encodedData)
        cFeature = self.gAR(encodedData)
        if self.quantizerContext is not None:
            cFeature = self.quantizerContext(cFeature)           
        return cFeature, encodedData, label, self.computeExtraLosses(batchData.device)

class CPCModelNullspace(nn.Module):

    def __init__(self,
                 cpc,
                 nullspace):

        super(CPCModelNullspace, self).__init__()
        self.cpc = cpc
        self.nullspace = nn.Linear(nullspace.shape[0], nullspace.shape[1], bias=False)
        self.nullspace.weight = nn.Parameter(nullspace.T)


    def forward(self, batchData, label):
        cFeature, encodedData, label = self.cpc(batchData, label)
        cFeature = self.nullspace(cFeature)
        encodedData = self.nullspace(encodedData)
        return cFeature, encodedData, label

class CPCModelPCA(nn.Module):
    def __init__(self,
                 cpc,
                 pcaA,
                 pcaB):

        super(CPCModelPCA, self).__init__()
        self.cpc = cpc
        self.pcaA = pcaA
        self.pcaB = pcaB


    def forward(self, batchData, label):
        cFeature, encodedData, label = self.cpc(batchData, label)
        cFeature[0] = cFeature[0] @ self.pcaA + self.pcaB
        encodedData = encodedData @ self.pcaA + self.pcaB
        return cFeature, encodedData, label

class ConcatenatedModel(nn.Module):

    def __init__(self, model_list):

        super(ConcatenatedModel, self).__init__()
        self.models = torch.nn.ModuleList(model_list)

    def forward(self, batchData, label):

        outFeatures = []
        outEncoded = []
        for model in self.models:
            cFeature, encodedData, label = model(batchData, label)
            outFeatures.append(cFeature)
            outEncoded.append(encodedData)
        return torch.cat(outFeatures, dim=2), \
            torch.cat(outEncoded, dim=2), label

class BoundaryPredictor(nn.Module):
    # 0.999998134144839 for 0.7 * 300 epochs
    # 0.999992163431717 for 50 epochs
    def __init__(self, featuresDim, nLayers=1, projRatio=2, lstmPolicy=True, temp=(2.0, 0.5, 0.999992163431717)):
        super(BoundaryPredictor, self).__init__()
        self.lstm = None
        if lstmPolicy:
            self.lstm = nn.LSTM(featuresDim + 1, featuresDim, num_layers=nLayers, batch_first=True)
            self.policy = nn.Sequential(
                nn.Linear(featuresDim, 2),
                # nn.LogSoftmax(dim=-1)
            )
        else:
            self.policy = nn.Sequential(
                nn.Linear(featuresDim + 1, featuresDim * projRatio),
                # nn.Linear(1, featuresDim * projRatio),
                nn.ReLU(),
                nn.Linear(featuresDim * projRatio, 2),
                # nn.LogSoftmax(dim=-1)
            )
            # self.policy = nn.Linear(1, 2)
        self.minSegmentWidth = 3
        self.numUpdates = 0
        self.maxTemp, self.minTemp, self.tempDecay = temp
        # self.AVG_BOUNDARY_DENSITY = 0.108697
        self.AVG_PHONE_DURATION = 7.580728 # 7.788017 # 8.931418
        # self.VAR_PHONE_DURATION = 3.587028 # 7.049214
        self.boundaryLogProbs = None
        # self.boundaryLogits = None
        # self.boundaryProbsPrior = None
        # self.ceLoss = torch.nn.CrossEntropyLoss(reduction='mean')

    def computeLengthPriorLoss(self):
        sampleLen = 8
        numSamples = 128
        # self.boundaryProbsPrior = self.boundaryProbsPrior.unsqueeze(2)
        # return 0.1 * self.ceLoss(
        #     self.boundaryLogits.view(-1, 2),
        #     torch.cat([1 - self.boundaryProbsPrior, self.boundaryProbsPrior], dim=-1).view(-1, 2), 
        # )
        # numSegmentsPriorLoss = torch.abs((self.boundaryLogProbs[:, :, 1].exp().sum() / self.boundaryLogProbs[:, :, 0].exp().sum()) - self.AVG_BOUNDARY_DENSITY)
        allSegmentsLenSampleProbs = F.unfold(self.boundaryLogProbs.view(-1, 2)[:, 1].exp().view(1, 1, -1, 1), (sampleLen, 1))
        selected = torch.randint(0, allSegmentsLenSampleProbs.size(-1), size=(numSamples,), )
        
        expectedNumBoundariesInSegsLenSample = allSegmentsLenSampleProbs[..., selected].sum(1)
        expectedValueLenLoss = (expectedNumBoundariesInSegsLenSample.mean() - (sampleLen / self.AVG_PHONE_DURATION))**2
        # varianceNumBoundariesInSegsLenSample = (allSegmentsLenSampleProbs[..., selected] * (1 - allSegmentsLenSampleProbs[..., selected])).sum(1)
        # varianceLenLoss = (varianceNumBoundariesInSegsLenSample.mean() - self.VAR_PHONE_DURATION)**2
        # expectedNumBoundariesInSegsLen8 = allSegmentsLen8.sum(1)
        # if self.training:
        #     coeff = max(-3.398e-5 * Globals.currentIteration + 1.0, 0.7)    # Linear annealing from 1.0 to 0.1 across 10 epochs
        # else:
        coeff = 1.0
        return coeff * expectedValueLenLoss

    def computeVariabilityPriorLoss(self):
        sampleLen = 32
        allSegmentsLenSampleProbs = F.unfold(self.boundaryLogProbs.view(-1, 2)[:, 1].exp().view(1, 1, -1, 1), (sampleLen, 1))
        expectedNBoundsInSegs = allSegmentsLenSampleProbs.sum(1)
        return -0.1 * torch.std(expectedNBoundsInSegs, dim=1, unbiased=False).mean()

    def computeEntropyLoss(self):
        entropy = -(self.boundaryLogProbs.view(-1, 2).exp() * self.boundaryLogProbs.view(-1, 2)).sum(-1).mean()
        if self.training:
            coeff = max(-(1/20) * Globals.epoch + 1.0, 0.25)    # Linear annealing from 1.0 to 0.1 across 20 epochs
        else:
            coeff = 0.25
        return -coeff * entropy
    
    def forward(self, x, label):
        bs, l, _ = x.size()
        self.boundaryLogProbs = torch.zeros((bs, l, 2), device=x.device)
        boundaries = torch.zeros((bs, l), device=x.device)
        prevBoundary = torch.zeros((bs, 1, 1), device=x.device)
        
        if self.lstm is not None:
            try:
                self.lstm.flatten_parameters()
            except RuntimeError:
                pass
            h = None
        noBoundaryFlag = torch.zeros((bs, 1), device=x.device)
        for i in range(l):
            inp = torch.cat((prevBoundary, x[:, i, :].view(bs, 1, -1)), dim=-1)
            if self.lstm is not None:
                lstmOut, h = self.lstm(inp, h)
                out = self.policy(lstmOut)
            else:
                out = self.policy(inp)
            tau = max(self.maxTemp * self.tempDecay ** self.numUpdates, self.minTemp)
            out[noBoundaryFlag.view(-1) > 0, :, 1] += -1e4
            out = torch.log_softmax(out / tau, dim=-1)   
            self.boundaryLogProbs[:, i, :] = out.view(bs, 2)
            if self.training:
                out = torch.multinomial(torch.exp(out.view(bs, -1)), 1)
            else:
                out = torch.argmax(out, dim=-1)
            noBoundaryFlag[noBoundaryFlag > 0] -= 1
            noBoundaryFlag[out > 0] = self.minSegmentWidth
            boundaries[:, i] = out.view(-1)
            prevBoundary = out.view(bs, 1, 1)
        return boundaries, torch.gather(self.boundaryLogProbs, -1, boundaries.unsqueeze(2).long())

EPS = 1e-7

class BiDiBoundaryPredictor(nn.Module):
    def __init__(self, featuresDim, nLayers=2, temp=(2.0, 0.5, 0.999992163431717)):
        super(BiDiBoundaryPredictor, self).__init__()
        from .transformers import buildTransformer
        self.transformer = buildTransformer(featuresDim, nLayers, 128, False, False)
        self.policy = nn.Sequential(
            nn.Linear(featuresDim, 2)
        )
        self.minSegmentWidth = 3
        self.numUpdates = 0
        self.maxTemp, self.minTemp, self.tempDecay = temp
        self.AVG_PHONE_DURATION = 7.580728 # 7.788017 # 8.931418
        self.boundaryLogProbs = None
        self.register_buffer('baseline', torch.Tensor([0]))
        self.EMA_COEFF = 0.99

    def computeLengthPriorLoss(self):
        sampleLen = 8
        numSamples = 128        
        allSegmentsLenSampleProbs = F.unfold(self.boundaryLogProbs.view(-1, 2)[:, 1].exp().view(1, 1, -1, 1), (sampleLen, 1))
        selected = torch.randint(0, allSegmentsLenSampleProbs.size(-1), size=(numSamples,), )
        
        expectedNumBoundariesInSegsLenSample = allSegmentsLenSampleProbs[..., selected].sum(1)
        expectedValueLenLoss = (expectedNumBoundariesInSegsLenSample.mean() - (sampleLen / self.AVG_PHONE_DURATION))**2
        # if self.training:
        #     coeff = max(-3.398e-5 * Globals.currentIteration + 1.0, 0.7)    # Linear annealing from 1.0 to 0.1 across 10 epochs
        # else:
        coeff = 1.0
        return coeff * expectedValueLenLoss

    def computeVariabilityPriorLoss(self):
        sampleLen = 32
        allSegmentsLenSampleProbs = F.unfold(self.boundaryLogProbs.view(-1, 2)[:, 1].exp().view(1, 1, -1, 1), (sampleLen, 1))
        expectedNBoundsInSegs = allSegmentsLenSampleProbs.sum(1)
        return -0.1 * torch.std(expectedNBoundsInSegs, dim=1, unbiased=False).mean()

    def computeEntropyLoss(self):
        entropy = -(self.boundaryLogProbs.view(-1, 2).exp() * self.boundaryLogProbs.view(-1, 2)).sum(-1).mean()
        if self.training:
            coeff = max(-(1/20) * Globals.epoch + 1.0, 0.25)    # Linear annealing from 1.0 to 0.1 across 20 epochs
        else:
            coeff = 0.25
        return -coeff * entropy
    
    def forward(self, x, label):
        bs, l, _ = x.size()
        x = self.transformer(x)
        x = self.policy(x)
        # # Ensuring minimum segment length
        # boundaries = torch.zeros((bs, l), device=x.device)        
        # noBoundaryFlag = torch.zeros((bs, 1), device=x.device)
        tau = max(self.maxTemp * self.tempDecay ** self.numUpdates, self.minTemp)
        # for i in range(l):
        #     x[noBoundaryFlag.view(-1) > 0, i, 1] += -1e4
        #     x[:, i, :] = torch.log_softmax(x[:, i, :] / tau, dim=-1)
        #     boundaryPreds = torch.argmax(x[:, i, :], dim=-1)
        #     noBoundaryFlag[noBoundaryFlag > 0] -= 1
        #     noBoundaryFlag[boundaryPreds > 0] = self.minSegmentWidth
        #     if self.training:
        #         boundaryPreds = torch.multinomial(torch.exp(x[:, i, :].view(bs, -1)), 1)
        #     boundaries[:, i] = boundaryPreds.view(-1)
        # self.boundaryLogProbs = x
        
        noBoundaryFlag = torch.zeros((bs, 1), device=x.device)
        for i in range(l):
            x[noBoundaryFlag.view(-1) > 0, i, 1] += -1e4
            boundaryPreds = torch.argmax(x[:, i, :], dim=-1)
            noBoundaryFlag[noBoundaryFlag > 0] -= 1
            noBoundaryFlag[boundaryPreds > 0] = self.minSegmentWidth

        self.boundaryLogProbs = torch.log_softmax(x / tau, dim=-1)
        if self.training:
            boundaries = torch.multinomial(torch.exp(self.boundaryLogProbs).view(bs * l, -1), 1).view(bs, l).float()
        else:
            boundaries = torch.argmax(self.boundaryLogProbs, dim=-1)
        return boundaries, torch.gather(self.boundaryLogProbs, -1, boundaries.unsqueeze(2).long())

    def computeSupervisedLoss(self, predictedBoundaries, label, toleranceInFrames=2):
        seqEndIdx = torch.arange(0, label.size(0)*label.size(1) + 1, label.size(1)).cuda()
        predictedBoundaries = torch.nonzero(predictedBoundaries.view(-1), as_tuple=True)[0]
        # Ensure that minibatch boundaries are preserved
        predictedBoundaries = torch.unique(torch.cat((predictedBoundaries, seqEndIdx)), sorted=True)
        diffs = torch.diff(label, dim=1)
        phone_changes = torch.cat((torch.ones((label.shape[0], 1), device=label.device), diffs), dim=1)
        trueBoundaries = torch.nonzero(phone_changes.contiguous().view(-1), as_tuple=True)[0]
        # Ensure that minibatch boundaries are preserved
        trueBoundaries = torch.unique(torch.cat((trueBoundaries, seqEndIdx)), sorted=True)

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

        reward = -torch.maximum(rVal, torch.Tensor([0]).cuda())
        firstIter = False
        if self.baseline.item() == 0:
            firstIter = True
            self.baseline += reward
        else:
            self.baseline *= self.EMA_COEFF
            self.baseline += ((1 - self.EMA_COEFF) * reward)

        Globals.writer.add_scalar(tag='REINFORCE baseline', scalar_value=self.baseline.item(), global_step=Globals.currentIteration)
        if firstIter:
            return (self.boundaryLogProbs.sum(1) * reward).mean()
        else:
            return (self.boundaryLogProbs.sum(1) * (reward - self.baseline)).mean()
        


EPS = 1e-12
DESIRED_SPECTRAL_RADIUS = 0.99

class Segmenter(nn.Module):

    def __init__(self,
                 segmentationMode,
                 segmentOnContext=False,
                 minNumSegments=2,
                 segmentCompression='average',
                 featuresDim=None,
                 nLayersBoundaryPredictor=1
                ):

        super(Segmenter, self).__init__()
        self.segmentOnContext = segmentOnContext
        self.segmentationMode = segmentationMode
        self.minNumSegments = minNumSegments
        assert featuresDim is not None or segmentCompression == 'average', "To use LSTM segment compression you need to pass the dimension of the features as argument"
        self.segmentCompression = segmentCompression
        if self.segmentCompression == 'lstm':
            self.featuresDim = featuresDim
            self.lstm = nn.LSTM(featuresDim, featuresDim, num_layers=1, batch_first=True)
            # self.lstm = nn.RNN(featuresDim, featuresDim, num_layers=1, batch_first=True)
            # # Random LSTM with echo state property
            # for param in self.lstm.parameters():
            #     param.requires_grad = False
            # weightHH = self.lstm.weight_hh_l0
            # spectralRadius = torch.max(torch.abs(torch.linalg.eig(weightHH)[0])).item()
            # self.lstm.weight_hh_l0 = torch.nn.Parameter((weightHH / spectralRadius) * DESIRED_SPECTRAL_RADIUS)
        self.boundaryPredictor = None
        if segmentationMode == "boundaryPredictor":
            # self.boundaryPredictor = BoundaryPredictor(featuresDim, nLayers=nLayersBoundaryPredictor)
            self.boundaryPredictor = BiDiBoundaryPredictor(featuresDim, nLayers=nLayersBoundaryPredictor)
        # self.boundaries = None
        # self.boundaryLogProbs = None

    # def computePGGTLoss(self, label, toleranceInFrames=2):
    #     diffs = torch.diff(label, dim=1)
    #     phoneChanges = torch.cat((torch.ones((label.shape[0], 1), device=label.device), diffs), dim=1)
    #     predErrors = torch.abs(torch.nonzero(self.boundaries.view(-1)) - torch.nonzero(phoneChanges.view(-1)).T)
    #     # True positives with 20 ms tolerance
    #     truePosIdxs = torch.nonzero(predErrors <= 2, as_tuple=True)[0]
    #     rewardsTruePos = torch.zeros((label.shape[0] * label.shape[1],), device=label.device)
    #     rewardsTruePos[truePosIdxs] = 1
    #     rewardsTruePos = rewardsTruePos.view(label.shape[0], label.shape[1])
    #     # 1 true detection at most matching to 1 prediction
    #     diffsTruePos = torch.diff(torch.cat((rewardsTruePos[:, 0].view(-1, 1), rewardsTruePos), 1))
    #     rewardsTruePos = torch.zeros((label.shape[0], label.shape[1],), device=label.device)
    #     rewardsTruePos[torch.nonzero(diffsTruePos > 0, as_tuple=True)] = 1

    #     # rewardsTruePos = torch.logical_and(phoneChanges, self.boundaries).float()
    #     rewardsTrueNegs = torch.logical_and(~phoneChanges.bool(), ~self.boundaries.bool()).float()
    #     # rewardsFalsePos = -(torch.logical_and(~phoneChanges, self.boundaries).float())
    #     # rewardsFalseNegs = -(torch.logical_and(phoneChanges, ~self.boundaries).float())
    #     reward = rewardsTruePos * 10 + rewardsTrueNegs # + rewardsFalsePos + 10 * rewardsFalseNegs
    #     return (-reward * self.boundaryLogProbs.view(label.size(0), label.size(1))).mean().view(1, 1)
    # #     trueBoundaries = torch.nonzero(diffs, as_tuple=True)
    # #     # Ensure that minibatch boundaries are preserved
    # #     # seqEndIdx = torch.arange(0, label.size(0) * label.size(1) + 1, label.size(1), device=label.device)
    # #     # trueBoundaries = torch.unique(torch.cat((trueBoundaries, seqEndIdx)), sorted=True)
    # #     predictedBoundaries = torch.nonzero(self.boundaries[:, 1:], as_tuple=True)
        
    # #     # precisionCounter = torch.zeros((len(torch.unique(predictedBoundaries[0])),), device=label.device)
    # #     # recallCounter = torch.zeros((len(torch.unique(trueBoundaries[0])),), device=label.device)
    # #     precisionCounter = torch.zeros((label.size(0),), device=label.device)
    # #     recallCounter = torch.zeros((label.size(0),), device=label.device)

    # #     for i, predictedBoundary in enumerate(predictedBoundaries[1]):
    # #         allTrue = trueBoundaries[1][trueBoundaries[0] == predictedBoundaries[0][i]]
    # #         if len(allTrue) > 0:
    # #             minDist = torch.min(torch.abs(allTrue - predictedBoundary))
    # #             precisionCounter[predictedBoundaries[0][i]] += (minDist <= toleranceInFrames)
            

    # #     for i, trueBoundary in enumerate(trueBoundaries[1]):
    # #         allPredicted = predictedBoundaries[1][predictedBoundaries[0] == trueBoundaries[0][i]]
    # #         if len(allPredicted) > 0:
    # #             minDist = torch.min(torch.abs(allPredicted - trueBoundary))
    # #             recallCounter[trueBoundaries[0][i]] += (minDist <= toleranceInFrames)

    # #     numPredicted = torch.zeros((label.size(0),), device=label.device)
    # #     batchsWPredictions, predsInBatchCount = torch.unique(predictedBoundaries[0], return_counts=True)
    # #     numPredicted[batchsWPredictions] = predsInBatchCount.float()
    # #     precision = precisionCounter / (numPredicted + EPS)
    # #     numBoundaries = torch.zeros((label.size(0),), device=label.device)
    # #     batchsWBoundaries, boundsInBatchCount = torch.unique(trueBoundaries[0], return_counts=True)
    # #     numBoundaries[batchsWBoundaries] = boundsInBatchCount.float()
    # #     recall = recallCounter / (numBoundaries + EPS)
    # #     # f1 = 2 * (precision * recall) / (precision + recall + EPS)
    # #     os = recall / (precision + EPS) - 1
    # #     r1 = torch.sqrt((1 - recall) ** 2 + os ** 2)
    # #     r2 = (-os + recall - 1) / (math.sqrt(2))
    # #     rVal = 1 - (torch.abs(r1) + torch.abs(r2)) / 2

    # #     return (-rVal.view(-1, 1, 1) * self.boundaryLogProbs).mean().view(1, 1)

    def computeExtraLosses(self, label):
        loss = {}
        if self.boundaryPredictor is not None:
            if Globals.supervisedRL:
                loss['supervisedRLLoss'] = self.boundaryPredictor.computeSupervisedLoss(self.predPhoneChanges, label).view(1, 1)
                loss['PhoneDensityPriorLoss'] = self.boundaryPredictor.computeLengthPriorLoss().view(1, 1)
                loss['SegmenterEntropyLoss'] = self.boundaryPredictor.computeEntropyLoss().view(1, 1)
            else:
                loss['PhoneDensityPriorLoss'] = self.boundaryPredictor.computeLengthPriorLoss().view(1, 1)
                # loss['VariablityPriorLoss'] = self.boundaryPredictor.computeVariabilityPriorLoss().view(1, 1)
                loss['SegmenterEntropyLoss'] = self.boundaryPredictor.computeEntropyLoss().view(1, 1)
        return loss

    def forward(self, cFeatures, encodedData, label, prominence=0.05, returnBoundaries=False): # , returnFlattened=False):
        # x = cFeatures if (self.segmentOnContext and not self.segmentationMode == 'boundaryPredictor') else encodedData
        x = cFeatures if self.segmentOnContext else encodedData
        bs, l, d = x.size()
        if self.segmentationMode == "cosineDissimilarity":
            from cpc.utils.misc import maxMinNorm
            from scipy.signal import find_peaks
            scores = F.cosine_similarity(x[:, :-1, :], x[:, 1:, :], dim=-1)
            scores = torch.cat([scores[:, 0].view(-1, 1), scores], dim=1)
            scores = 1 - maxMinNorm(scores)
            boundaries, _ = find_peaks(scores.view(-1).cpu().detach().numpy(), prominence=prominence)
            if len(boundaries) == 0:
                boundaries = [0]
            boundaries = torch.tensor(boundaries, device=x.device)
        elif self.segmentationMode == "collapseRepetitions":
            pSoft = torch.cat((torch.ones(bs, 1, d).to(x.device), 
                            torch.abs(torch.diff(x, dim=1))), dim=1).sum(2)
            pHard = (pSoft > 0) * 1.0
            boundaries = torch.where(pHard.view(-1))[0]
            seqEndIdx = torch.arange(0, bs * l + 1, l, device=x.device)
            boundaries = torch.unique(torch.cat((boundaries, seqEndIdx)), sorted=True)
        elif self.segmentationMode == "boundaryPredictor":
            predPhoneChanges, boundaryLogProbs = self.boundaryPredictor(encodedData, label)
            self.predPhoneChanges = predPhoneChanges
            boundaries =  torch.nonzero(predPhoneChanges.view(-1), as_tuple=True)[0]
            # self.boundaryLogProbs = boundaryLogProbs
            # self.boundaries = predPhoneChanges.bool()
        elif self.segmentationMode.startswith('groundTruth'):
            assert label is not None, "To use ground truth segmentation labels must be provided"
            diffs = torch.diff(label, dim=1)
            phoneChanges = torch.cat((torch.ones((bs, 1)).to(x.device), diffs), dim=1)
            if self.segmentationMode in ['groundTruthNumSegments', 'groundTruthUnderMixed', 'groundTruthOverMixed']:
                numSegments = (phoneChanges != 0).sum(-1)
                boundaries = []
                if self.segmentationMode == 'groundTruthUnderMixed':
                    underSampleFactor = 2 
                elif self.segmentationMode == 'groundTruthOverMixed':
                    underSampleFactor = 0.5
                else:
                    underSampleFactor = 1                
                for b in range(bs):
                    boundaries.append(b * l + torch.round(torch.arange(0, l, l / (numSegments[b] / underSampleFactor), 
                    device=cFeatures.device)).int())
                boundaries = torch.cat(boundaries)
            else:
                boundaries = torch.nonzero(phoneChanges.contiguous().view(-1), as_tuple=True)[0]
            if self.segmentationMode == 'groundTruthWError':
                maxOffset = 2
                origBoundaries = boundaries[boundaries % x.size(1) != 0]
                noiseOffset = torch.randint_like(origBoundaries, low=-maxOffset, high=maxOffset + 1)
                newBoundaries = origBoundaries + noiseOffset
                toFix = torch.where((origBoundaries // x.size(1) != newBoundaries // x.size(1)) | (newBoundaries < 0))[0]
                newBoundaries[toFix] = origBoundaries[toFix]
                boundaries = newBoundaries
            elif self.segmentationMode == 'groundTruthUnder':
                subsamplingFactor = 4
                boundaries = boundaries[boundaries % x.size(1) != 0]
                perm = torch.randperm(boundaries.size(0))
                idx = perm[:boundaries.size(0) // subsamplingFactor]
                boundaries = boundaries[idx]
            elif self.segmentationMode == 'groundTruthOver':
                oversamplingFactor = 4
                for _ in range(int(math.log2(oversamplingFactor))):
                    addedBoundaries = torch.clone(boundaries)[:-1]
                    addedBoundaries = addedBoundaries + (torch.diff(boundaries) // 2)
                    boundaries = torch.unique(torch.cat((boundaries, addedBoundaries)), sorted=True)
        else:
            raise NotImplementedError
        if returnBoundaries:
            predPhoneChanges = torch.zeros((bs * l,), device=boundaries.device)
            predPhoneChanges[boundaries] = 1
            return predPhoneChanges.view(bs, l)
        # Ensure that minibatch boundaries are preserved
        seqEndIdx = torch.arange(0, x.size(0)*x.size(1) + 1, x.size(1), device=x.device)
        boundaries = torch.unique(torch.cat((boundaries, seqEndIdx)), sorted=True)
        
        compressMatrices, compressedLens = getCompressionMatrix(boundaries, 
            x.size(1), 
            x.device,
            0 if (self.segmentCompression == 'lstm' or not self.training) else self.minNumSegments,
            randomPool=self.segmentCompression == 'random'
        )
        if self.segmentCompression in ['average', 'random']:
            # Get averaged segments
            assert compressMatrices.shape[0] == x.shape[0]
            packedCompressedX = compressBatch(
                x, compressMatrices, compressedLens, pack=True
            )
        elif self.segmentCompression == 'lstm':
            segmentLens = compressMatrices.sum(-1).view(-1).int()
            segmentLens = segmentLens[segmentLens != 0]
            x = x.reshape(-1, d)
            x = torch.split(x, tuple(segmentLens.int()))
            
            if self.training:
                # Upsampling each segment so that all segments have the same length 
                # (intended to avoid RNN cheating by counting):
                x = list(x)
                upsampledLen = max(segmentLens).item()
                for i in range(len(x)):
                    x[i] = torch.nn.functional.interpolate(
                        x[i].view(1, segmentLens[i].item(), d).permute(0, 2, 1),
                        size=upsampledLen, mode='linear').permute(0, 2, 1)
                x = torch.vstack(x)
            else:
                x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
            try:
                self.lstm.flatten_parameters()
            except RuntimeError:
                pass
            _, (x, _) = self.lstm(x)
            # _, x = self.lstm(x)
            x = torch.split(x.view(-1, d), tuple(compressedLens))
            if (compressedLens < self.minNumSegments).any() and self.training:
                segments = []
                for compressedSegment, seqLen in zip(x, compressedLens):
                    if seqLen < self.minNumSegments:
                        compressedSegment = F.pad(
                            compressedSegment.unsqueeze(0), (0, 0, 0, (self.minNumSegments - seqLen).int()), 
                            mode='replicate'
                        ).squeeze()
                    segments.append(compressedSegment)
            else:
                segments = x
            packedCompressedX = torch.nn.utils.rnn.pack_sequence(segments, enforce_sorted=False)
        # if returnBoundaries:            
        #     return packedCompressedX, boundaries
        # else:
        if self.segmentationMode == 'boundaryPredictor':
            boundaryLogProbs = compressBatch(
                boundaryLogProbs, compressMatrices, compressedLens, pack=True, average=False
            )
        return packedCompressedX, compressMatrices, boundaryLogProbs if self.segmentationMode == 'boundaryPredictor' else None


class MultiLevelModel(nn.Module):

    def __init__(self,
                 frameLevelModel,
                 segmenter=None,
                 keepHidden=False
                ):

        super(MultiLevelModel, self).__init__()
        self.frameLevelModel = frameLevelModel
        self.segmenter = segmenter
        inputDim =  frameLevelModel.gAR.getDimOutput() if (segmenter is not None and segmenter.segmentOnContext) else frameLevelModel.gEncoder.dimEncoded
        if Globals.uniformDownsampling:
            self.sEncoder = nn.Sequential(
                nn.Conv1d(inputDim, 256, 8, 8),
                nn.ReLU()
            )
        else:
            self.sEncoder = nn.Sequential(
                nn.Linear(inputDim, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )
        self.sAR = nn.LSTM(256, 256, num_layers=1, batch_first=True)
        self.keepHidden = keepHidden
        self.hidden = None
        self.numUpdates = 0

    def updateCounter(self):
        self.numUpdates += 1
        if self.segmenter is not None:
            if self.segmenter.boundaryPredictor is not None:
                self.segmenter.boundaryPredictor.numUpdates += 1
        self.frameLevelModel.updateCounter()

    def forward(self, batchData, label, upsampleSegments=False):
        cFeature, encodedData, label, extraLosses = self.frameLevelModel(batchData, label)
        boundaryLogProbs = None
        if self.segmenter is not None:
            compressedSegments, compressMatrices, boundaryLogProbs = self.segmenter(cFeature, encodedData, label)
            segmenterLosses = self.segmenter.computeExtraLosses(label)
            extraLosses.update(segmenterLosses)
            encodedSegments = torch.nn.utils.rnn.PackedSequence(
                self.sEncoder(compressedSegments.data),
                compressedSegments.batch_sizes, compressedSegments.sorted_indices, compressedSegments.unsorted_indices
            )
        else:
            if Globals.uniformDownsampling:
                encodedSegments = self.sEncoder(encodedData.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                encodedSegments = self.sEncoder(encodedData)
        try:
            self.sAR.flatten_parameters()
        except RuntimeError:
            pass
        segmentsContext, hidden = self.sAR(encodedSegments, self.hidden)    

        if self.keepHidden:            
            self.hidden = tuple(x.detach() for x in hidden)

        if isinstance(segmentsContext, torch.nn.utils.rnn.PackedSequence):
            # We pad to have the maximum possible sequence length so as to be compatible with multi GPU setup
            paddedContextSegments, segmentSeqLens = torch.nn.utils.rnn.pad_packed_sequence(
                segmentsContext, batch_first=True, total_length=cFeature.size(1)
            )
            paddedEncodedSegments, segmentSeqLens = torch.nn.utils.rnn.pad_packed_sequence(
                encodedSegments, batch_first=True, total_length=cFeature.size(1)
            )
        else:
            paddedContextSegments = segmentsContext
            paddedEncodedSegments = encodedSegments
            if Globals.uniformDownsampling:
                segmentSeqLens = torch.IntTensor([paddedEncodedSegments.size(1)] * paddedEncodedSegments.size(0))
            else:
                segmentSeqLens = torch.IntTensor([encodedData.size(1)] * encodedData.size(0))
        # segmentLens = compressMatrices.sum(-1)
        # paddedSegmentLens = F.pad(segmentLens, (0, cFeature.size(1) - segmentLens.size(1)))      
        
        if Globals.supervisedRL:
            boundaryLogProbs = None

        if boundaryLogProbs is not None:
            # extraLosses['RValPGLoss'] = self.segmenter.computePGGTLoss(label)
            boundaryLogProbs, segmentSeqLens = torch.nn.utils.rnn.pad_packed_sequence(
                boundaryLogProbs, batch_first=True, total_length=cFeature.size(1)
            )

        if upsampleSegments:
            # if self.segmenter is not None:    
            upsampledBatchContext = []
            upsampledBatchEncodings = []
            segmentLens = compressMatrices.sum(-1).int()
            for b in range(paddedContextSegments.size(0)):
                segmentContext = paddedContextSegments[b, :segmentSeqLens[b], :]
                segmentEncodings = paddedEncodedSegments[b, :segmentSeqLens[b], :]
                upsampledBatchContext.append(segmentContext.repeat_interleave(segmentLens[b, :segmentSeqLens[b]], dim=0))
                upsampledBatchEncodings.append(segmentEncodings.repeat_interleave(segmentLens[b, :segmentSeqLens[b]], dim=0))
            segmentsContext = torch.stack(upsampledBatchContext)
            encodedSegments = torch.stack(upsampledBatchEncodings)
        else:
            segmentsContext = {
                'paddedCFeatures': paddedContextSegments,
                'segmentSeqLens': segmentSeqLens.to(paddedContextSegments.device),
                'boundaryLogProbs': boundaryLogProbs,
                # 'segmentLens': paddedSegmentLens
            }
            encodedSegments = paddedEncodedSegments
        return [cFeature, segmentsContext], [encodedData, encodedSegments], label, extraLosses