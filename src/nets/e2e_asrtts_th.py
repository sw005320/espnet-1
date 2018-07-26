#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging
import sys

import chainer
from chainer import reporter
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from e2e_asr_attctc_th import to_cuda
from e2e_asr_attctc_th import pad_list
from e2e_tts_th import make_mask

from tts_pytorch import pad_ndarray_list

torch_is_old = torch.__version__.startswith("0.3.")


class Reporter(chainer.Chain):
    def report(self, loss, asr_loss, tts_loss, s2s_loss, t2t_loss, asr_acc, t2t_acc):
        if loss:
            reporter.report({'loss': loss}, self)
        if asr_loss:
            reporter.report({'asr_loss': asr_loss}, self)
        if tts_loss:
            reporter.report({'tts_loss': tts_loss}, self)
        if s2s_loss:
            reporter.report({'s2s_loss': s2s_loss}, self)
        if t2t_loss:
            reporter.report({'t2t_loss': t2t_loss}, self)
        if asr_acc:
            reporter.report({'asr_acc': asr_acc}, self)
        if t2t_acc:
            reporter.report({'t2t_acc': t2t_acc}, self)


class ASRTTSLoss(torch.nn.Module):
    def __init__(self, asr_loss, tts_loss, args, return_targets=True):
        super(ASRTTSLoss, self).__init__()
        self.asr_loss = asr_loss
        self.tts_loss = tts_loss

        self.ae_speech = AutoEncoderSpeech(asr_loss, tts_loss, args)
        self.ae_text = AutoEncoderText(asr_loss, tts_loss, args)

        self.reporter = Reporter()
        self.return_targets = return_targets

    def forward(self, data):
        # these are used to adujst the loss and also pass the data to tts
        tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens = get_tts_data(self, data, 'text')
        avg_featlen = float(np.mean(tts_featlens.data.cpu().numpy()))
        if data[0][1]['utt2mode'] == 'p':
            logging.info("parallel data mode")
            asr_loss, asr_acc = self.asr_loss(data, do_report=False, report_acc=True)  # disable reporter
            tts_loss = self.tts_loss(tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens, do_report=False)
            s2s_loss = self.ae_speech(data)
            t2t_loss, t2t_acc = self.ae_text(data)
            loss = asr_loss + avg_featlen * tts_loss + avg_featlen * s2s_loss + t2t_loss
            self.reporter.report(loss, asr_loss, tts_loss, s2s_loss, t2t_loss, asr_acc, t2t_acc)
        elif data[0][1]['utt2mode'] == 'a':
            logging.info("audio only mode")
            s2s_loss = self.ae_speech(data)
            loss = avg_featlen * s2s_loss
            self.reporter.report(loss, None, None, s2s_loss, None, None, None)
        elif data[0][1]['utt2mode'] == 't':
            logging.info("text only mode")
            t2t_loss, t2t_acc = self.ae_text(data)
            loss = t2t_loss
            self.reporter.report(loss, None, None, None, t2t_loss, None, t2t_acc)
        else:
            logging.error("Error: cannot find correct mode ('p', 'a', 't')")
            sys.exit()

        return loss


def get_asr_data(self, data, sort_by):
    # utt list of frame x dim
    xs = [d[1]['feat'] for d in data]
    # remove 0-output-length utterances
    tids = [d[1]['output'][0]['tokenid'].split() for d in data]
    filtered_index = filter(lambda i: len(tids[i]) > 0, range(len(xs)))
    if sort_by == 'feat':
        sorted_index = sorted(filtered_index, key=lambda i: -len(xs[i]))
    elif sort_by == 'text':
        sorted_index = sorted(filtered_index, key=lambda i: -len(tids[i]))
    else:
        logging.error("Error: specify 'text' or 'feat' to sort")
        sys.exit()
    if len(sorted_index) != len(xs):
        logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
            len(xs), len(sorted_index)))
    xs = [xs[i] for i in sorted_index]
    # utt list of olen
    texts = [np.fromiter(map(int, tids[i]), dtype=np.int64) for i in sorted_index]
    if torch_is_old:
        texts = [to_cuda(self, Variable(torch.from_numpy(y), volatile=not self.training)) for y in texts]
    else:
        texts = [to_cuda(self, torch.from_numpy(y)) for y in texts]

    # subsample frame
    xs = [xx[::self.subsample[0], :] for xx in xs]
    featlens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
    if torch_is_old:
        hs = [to_cuda(self, Variable(torch.from_numpy(xx), volatile=not self.training)) for xx in xs]
    else:
        hs = [to_cuda(self, torch.from_numpy(xx)) for xx in xs]

    # 1. encoder
    feats = pad_list(hs, 0.0)

    return texts, feats, featlens


def get_tts_data(self, data, sort_by):
    # get eos
    eos = str(int(data[0][1]['output'][0]['shape'][1]) - 1)

    # get target features and input character sequence
    texts = [b[1]['output'][0]['tokenid'].split() + [eos] for b in data]
    feats = [b[1]['feat'] for b in data]

    # remove empty sequence and get sort along with length
    filtered_idx = filter(lambda i: len(texts[i]) > 0, range(len(feats)))
    if sort_by == 'feat':
        sorted_idx = sorted(filtered_idx, key=lambda i: -len(feats[i]))
    elif sort_by == 'text':
        sorted_idx = sorted(filtered_idx, key=lambda i: -len(texts[i]))
    else:
        logging.error("Error: specify 'text' or 'feat' to sort")
        sys.exit()
    texts = [np.fromiter(map(int, texts[i]), dtype=np.int64) for i in sorted_idx]
    feats = [feats[i] for i in sorted_idx]

    # get list of lengths (must be tensor for DataParallel)
    textlens = torch.from_numpy(np.fromiter((x.shape[0] for x in texts), dtype=np.int64))
    featlens = torch.from_numpy(np.fromiter((y.shape[0] for y in feats), dtype=np.int64))

    # perform padding and convert to tensor
    texts = torch.from_numpy(pad_ndarray_list(texts, 0)).long()
    feats = torch.from_numpy(pad_ndarray_list(feats, 0)).float()

    # make labels for stop prediction
    labels = feats.new(feats.size(0), feats.size(1)).zero_()
    for i, l in enumerate(featlens):
        labels[i, l - 1:] = 1

    if torch_is_old:
        texts = to_cuda(self, texts, volatile=not self.training)
        feats = to_cuda(self, feats, volatile=not self.training)
        labels = to_cuda(self, labels, volatile=not self.training)
    else:
        texts = to_cuda(self, texts)
        feats = to_cuda(self, feats)
        labels = to_cuda(self, labels)

    if self.return_targets:
        return texts, textlens, feats, labels, featlens
    else:
        return texts, textlens, feats


def get_subsample(args):
    subsample = np.ones(args.elayers + 1, dtype=np.int)
    if args.etype == 'blstmp':
        ss = args.subsample.split("_")
        for j in range(min(args.elayers + 1, len(ss))):
            subsample[j] = int(ss[j])
    else:
        logging.warning('Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
    logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))

    return subsample


class AutoEncoderSpeech(torch.nn.Module):
    def __init__(self, asr_loss, tts_loss, args, return_targets=True):
        super(AutoEncoderSpeech, self).__init__()
        self.asr_loss = asr_loss
        self.tts_loss = tts_loss
        self.asr_enc = asr_loss.predictor.enc
        self.tts_dec = tts_loss.model.dec
        self.subsample = get_subsample(args)
        self.return_targets = return_targets

    def forward(self, data):
        asr_texts, asr_feats, asr_featlens = get_asr_data(self, data, 'feat')
        tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens = get_tts_data(self, data, 'feat')

        # encoder
        hpad, hlens = self.asr_enc(asr_feats, asr_featlens)

        after_outs, before_outs, logits = self.tts_dec(hpad, hlens, tts_feats)
        # copied from e2e_tts_th.py
        if self.tts_loss.use_masking and tts_featlens is not None:
            # weight positive samples
            if self.tts_loss.bce_pos_weight != 1.0:
                # TODO(kan-bayashi): need to be fixed in pytorch v4
                weights = tts_feats.data.new(*tts_labels.size()).fill_(1)
                if torch_is_old:
                    weights = Variable(weights, volatile=tts_feats.volatile)
                weights.masked_fill_(tts_labels.eq(1), self.tts_loss.bce_pos_weight)
            else:
                weights = None
            # masking padded values
            mask = make_mask(tts_featlens, tts_feats.size(2))
            if torch.cuda.is_available():
                tts_feats = tts_feats.cuda()
                after_outs = after_outs.cuda()
                before_outs = before_outs.cuda()
                tts_labels = tts_labels.cuda()
                logits = logits.cuda()
                if weights is not None:
                    weights = weights.cuda()
            feats = tts_feats.masked_select(mask)
            after_outs = after_outs.masked_select(mask)
            before_outs = before_outs.masked_select(mask)
            labels = tts_labels.masked_select(mask[:, :, 0])
            logits = logits.masked_select(mask[:, :, 0])
            weights = weights.masked_select(mask[:, :, 0]) if weights is not None else None
            # calculate loss
            l1_loss = F.l1_loss(after_outs, feats) + F.l1_loss(before_outs, feats)
            mse_loss = F.mse_loss(after_outs, feats) + F.mse_loss(before_outs, feats)
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels, weights)
            loss = l1_loss + mse_loss + bce_loss
        else:
            # calculate loss
            l1_loss = F.l1_loss(after_outs, tts_feats) + F.l1_loss(before_outs, tts_feats)
            mse_loss = F.mse_loss(after_outs, tts_feats) + F.mse_loss(before_outs, tts_feats)
            bce_loss = F.binary_cross_entropy_with_logits(logits, tts_labels)
            loss = l1_loss + mse_loss + bce_loss

        # report loss values for logging
        loss_data = loss.data[0] if torch_is_old else loss.item()
        l1_loss_data = l1_loss.data[0] if torch_is_old else l1_loss.item()
        bce_loss_data = bce_loss.data[0] if torch_is_old else bce_loss.item()
        mse_loss_data = mse_loss.data[0] if torch_is_old else mse_loss.item()
        logging.debug("loss = %.3e (bce: %.3e, l1: %.3e, mse: %.3e)" % (
            loss_data, bce_loss_data, l1_loss_data, mse_loss_data))
        # self.tts_loss.reporter.report(l1_loss_data, mse_loss_data, bce_loss_data, loss_data)

        return loss


class AutoEncoderText(torch.nn.Module):
    def __init__(self, asr_loss, tts_loss, args, return_targets=True):
        super(AutoEncoderText, self).__init__()
        self.asr_loss = asr_loss
        self.tts_loss = tts_loss
        self.tts_enc = tts_loss.model.enc
        self.asr_dec = asr_loss.predictor.dec
        self.subsample = get_subsample(args)
        self.return_targets = return_targets

    def forward(self, data):
        asr_texts, asr_feats, asr_featlens = get_asr_data(self, data, 'text')
        tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens = get_tts_data(self, data, 'text')
        
        if isinstance(tts_textlens, torch.Tensor) or isinstance(tts_textlens, np.ndarray):
            tts_textlens = list(map(int, tts_textlens))

        hpad, hlens = self.tts_enc(tts_texts, tts_textlens)

        # NOTE asr_texts and tts_texts would be different due to the <eos> treatment
        loss, acc = self.asr_dec(hpad, hlens, asr_texts)

        return loss, acc
