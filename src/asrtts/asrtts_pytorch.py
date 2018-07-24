#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import json
import logging
import math
import numpy as np
import random
import os
import pickle
import sys

# chainer related
import chainer

from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions

# torch related
import torch

# spnet related
from asr_utils import adadelta_eps_decay
from asr_utils import CompareValueTrigger
from asr_utils import converter_kaldi
from asr_utils import delete_feat
from asr_utils import load_labeldict
from asr_utils import make_batchset
from asr_utils import PlotAttentionReport
from asr_utils import restore_snapshot


from e2e_asr_attctc_th import E2E
from e2e_asr_attctc_th import Loss
from e2e_asr_attctc_th import torch_is_old

from e2e_tts_th import Tacotron2
from e2e_tts_th import Tacotron2Loss

from e2e_asrtts_th import ASRTTSLoss
from e2e_asrtts_th import get_tts_data
# for kaldi io
import kaldi_io_py

# rnnlm
import extlm_pytorch
import lm_pytorch

# matplotlib related
import matplotlib
matplotlib.use('Agg')


class CustomEvaluater(extensions.Evaluator):
    '''Custom evaluater for pytorch'''

    def __init__(self, model, iterator, target, converter, device):
        super(CustomEvaluater, self).__init__(
            iterator, target, converter=converter, device=device)
        self.model = model

    # The core part of the update routine can be customized by overriding.
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                # read scp files
                # x: original json with loaded features
                #    will be converted to chainer variable later
                data = self.converter(batch)
                self.model.eval()

                if data[0][1]['utt2mode'] != 'p':
                    logging.error("Error: evaluation only support a parallel data mode ('p')")
                    sys.exit()
                tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens = get_tts_data(self.model, data, 'text')
                avg_featlen = float(np.mean(tts_featlens.data.cpu().numpy()))

                self.model.asr_loss(data, do_report=False, report_acc=True)
                asr_loss, asr_acc = self.model.asr_loss(data, do_report=False, report_acc=True)  # disable reporter
                loss = asr_loss
                self.model.reporter.report(loss, asr_loss, None, None, None, asr_acc, None)
                loss.detach()

                tts_loss = self.model.tts_loss(tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens,
                                               do_report=False)
                loss = avg_featlen * tts_loss
                self.model.reporter.report(loss, None, tts_loss, None, None, None, None)
                loss.detach()

                s2s_loss = self.model.ae_speech(data)
                loss = avg_featlen * s2s_loss
                self.model.reporter.report(loss, None, None, s2s_loss, None, None, None)
                loss.detach()

                t2t_loss, t2t_acc = self.model.ae_text(data)
                loss = t2t_loss
                self.model.reporter.report(loss, None, None, None, t2t_loss, None, t2t_acc)
                loss.detach()

                delete_feat(data)

            summary.add(observation)

        self.model.train()

        return summary.compute_mean()


def update_parameters(att, true_or_false):
    for child in att.children():
        for param in child.parameters():
            param.requires_grad = true_or_false


class CustomUpdater(training.StandardUpdater):
    '''Custom updater for pytorch'''

    def __init__(self, model, grad_clip_threshold, train_iter,
                 optimizer, converter, device):
        super(CustomUpdater, self).__init__(
            train_iter, optimizer, converter=converter, device=None)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.num_gpu = len(device)

        if torch_is_old:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm
        else:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm_

    def gradient_decent(self, loss, optimizer):
        optimizer.zero_grad()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.detach()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = self.clip_grad_norm(self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.__next__()

        # read scp files
        # x: original json with loaded features
        #    will be converted to chainer variable later
        # batch only has one minibatch utterance, which is specified by batch[0]
        if len(batch[0]) < self.num_gpu:
            logging.warning('batch size is less than number of gpus. Ignored')
            return
        data = self.converter(batch)

        # Compute the loss at this time step and accumulate it
        tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens = get_tts_data(self.model, data, 'text')
        avg_featlen = float(np.mean(tts_featlens.data.cpu().numpy()))
        if data[0][1]['utt2mode'] == 'p':
            logging.info("parallel data mode")
            modes = ['asr', 'tts', 's2s', 't2t']
            random.shuffle(modes)
            # shuffle
            for mode in modes:
                if mode == 'asr':
                    asr_loss, asr_acc = self.model.asr_loss(data, do_report=False, report_acc=True)  # disable reporter
                    loss = asr_loss
                    self.model.reporter.report(loss, asr_loss, None, None, None, asr_acc, None)
                if mode == 'tts':
                    tts_loss = self.model.tts_loss(tts_texts, tts_textlens, tts_feats, tts_labels, tts_featlens, do_report=False)
                    loss = avg_featlen * tts_loss
                    self.model.reporter.report(loss, None, tts_loss, None, None, None, None)
                if mode == 's2s':
                    update_parameters(self.model.tts_loss.model.dec.att, False)
                    s2s_loss = self.model.ae_speech(data)
                    loss = avg_featlen * s2s_loss
                    self.model.reporter.report(loss, None, None, s2s_loss, None, None, None)
                if mode == 't2t':
                    update_parameters(self.model.asr_loss.predictor.att, False)
                    t2t_loss, t2t_acc = self.model.ae_text(data)
                    loss = t2t_loss
                    self.model.reporter.report(loss, None, None, None, t2t_loss, None, t2t_acc)
                self.gradient_decent(loss, optimizer)
                update_parameters(self.model.asr_loss.predictor.att, True)
                logging.info("loss: %f", loss[0].data[0] if torch_is_old else loss[0].item())
        elif data[0][1]['utt2mode'] == 'a':
            logging.info("audio only mode")
            s2s_loss = self.model.ae_speech(data)
            loss = avg_featlen * s2s_loss
            self.gradient_decent(loss, optimizer)
            self.model.reporter.report(loss, None, None, s2s_loss, None, None, None)
        elif data[0][1]['utt2mode'] == 't':
            logging.info("text only mode")
            t2t_loss, t2t_acc = self.model.ae_text(data)
            loss = t2t_loss
            self.gradient_decent(loss, optimizer)
            self.model.reporter.report(loss, None, None, None, t2t_loss, None, t2t_acc)
        else:
            logging.error("Error: cannot find correct mode ('p', 'a', 't')")
            sys.exit()

        delete_feat(data)


class DataParallel(torch.nn.DataParallel):
    def scatter(self, inputs, kwargs, device_ids, dim):
        r"""Scatter with support for kwargs dictionary"""
        if len(inputs) == 1:
            inputs = inputs[0]
        avg = int(math.ceil(len(inputs) * 1. / len(device_ids)))
        # inputs = scatter(inputs, device_ids, dim) if inputs else []
        inputs = [[inputs[i:i + avg]] for i in range(0, len(inputs), avg)]
        kwargs = torch.nn.scatter(kwargs, device_ids, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids, self.dim)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)


def extract_json(json, mode):
    all_utt = {}
    for utt in json:
        if json[utt]['utt2mode'] == mode:
            all_utt[utt]= json[utt]

    return all_utt


def make_batchset_asrtts(data, batch_size, max_length_in, max_length_out, num_batches=0, factor_audio=1, factor_text=5):

    data_audio_only = extract_json(data, 'a')
    data_text_only = extract_json(data, 't')
    data_parallel = extract_json(data, 'p')

    if len(data_audio_only) > 0:
        json_audio_only = make_batchset(data_audio_only, factor_audio * batch_size,
                                        max_length_in, max_length_out, num_batches)
    else:
        json_audio_only = []
    if len(data_text_only) > 0:
        json_text_only = make_batchset(data_text_only, factor_text * batch_size,
                                       max_length_in, max_length_out, num_batches)
    else:
        json_text_only = []
    if len(data_parallel) > 0:
        json_parallel = make_batchset(data_parallel, batch_size, max_length_in,
                                      max_length_out, num_batches)
    else:
        json_parallel = []

    return json_audio_only + json_text_only + json_parallel


def train(args):
    '''Run training'''
    # seed setting
    torch.manual_seed(args.seed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # revmoe type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info('torch type check is disabled')
    # use determinisitic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        logging.info('torch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    # TODO(nelson) remove in future
    if 'input' not in valid_json[utts[0]]:
        logging.error(
            "input file format (json) is modified, please redo"
            "stage 2: Dictionary and Json Data Preparation")
        sys.exit(1)
    idim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    # specify model architecture for ASR
    e2e_asr = E2E(idim, odim, args)
    asr_loss = Loss(e2e_asr, args.mtlalpha)

    # define output activation function
    if args.tts_output_activation is None:
        output_activation_fn = None
    elif hasattr(torch.nn.functional, args.tts_output_activation):
        output_activation_fn = getattr(torch.nn.functional, args.tts_output_activation)
    else:
        raise ValueError('there is no such an activation function. (%s)' % args.tts_output_activation)

    # specify model architecture for TTS
    # reverse input and output dimension
    e2e_tts = Tacotron2(
        idim=odim,
        odim=idim,
        embed_dim=args.tts_embed_dim,
        elayers=args.tts_elayers,
        eunits=args.tts_eunits,
        econv_layers=args.tts_econv_layers,
        econv_chans=args.tts_econv_chans,
        econv_filts=args.tts_econv_filts,
        dlayers=args.tts_dlayers,
        dunits=args.tts_dunits,
        prenet_layers=args.tts_prenet_layers,
        prenet_units=args.tts_prenet_units,
        postnet_layers=args.tts_postnet_layers,
        postnet_chans=args.tts_postnet_chans,
        postnet_filts=args.tts_postnet_filts,
        output_activation_fn=output_activation_fn,
        adim=args.tts_adim,
        aconv_chans=args.tts_aconv_chans,
        aconv_filts=args.tts_aconv_filts,
        cumulate_att_w=args.tts_cumulate_att_w,
        use_batch_norm=args.tts_use_batch_norm,
        use_concate=args.tts_use_concate,
        dropout=args.tts_dropout_rate,
        zoneout=args.tts_zoneout_rate)
    logging.info(e2e_tts)
    tts_loss = Tacotron2Loss(
        model=e2e_tts,
        use_masking=args.tts_use_masking,
        bce_pos_weight=args.tts_bce_pos_weight)

    # define loss
    model = ASRTTSLoss(asr_loss, tts_loss, args)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.conf'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        # TODO(watanabe) use others than pickle, possibly json, and save as a text
        pickle.dump((idim, odim, args), f)
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # Set gpu
    reporter = model.reporter
    ngpu = args.ngpu
    if ngpu == 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
    elif ngpu > 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model = torch.nn.DataParallel(model, device_ids=gpu_id)
        model.cuda()
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu
    else:
        gpu_id = [-1]

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     args.lr, eps=args.eps,
                                     weight_decay=args.weight_decay)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train = make_batchset_asrtts(train_json, args.batch_size,
                                 args.maxlen_in, args.maxlen_out, args.minibatches)
    valid = make_batchset_asrtts(valid_json, args.batch_size,
                                 args.maxlen_in, args.maxlen_out, args.minibatches)
    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    train_iter = chainer.iterators.SerialIterator(train, 1)
    valid_iter = chainer.iterators.SerialIterator(
        valid, 1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = CustomUpdater(model, args.grad_clip, train_iter, optimizer,
                            converter=converter_kaldi, device=gpu_id)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
        if ngpu > 1:
            model.module.load_state_dict(torch.load(args.outdir + '/model.acc.best'))
        else:
            model.load_state_dict(torch.load(args.outdir + '/model.acc.best'))
        model = trainer.updater.model

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluater(
        model, valid_iter, reporter, converter=converter_kaldi, device=gpu_id))

    # Save attention weight each epoch
    '''
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        data = converter_kaldi([data], device=gpu_id)
        trainer.extend(PlotAttentionReport(model, data, args.outdir + "/att_ws"), trigger=(1, 'epoch'))
    '''

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/asr_loss', 'validation/main/asr_loss',
                                          'main/tts_loss', 'validation/main/tts_loss',
                                          'main/s2s_loss', 'validation/main/s2s_loss',
                                          'main/t2t_loss', 'validation/main/t2t_loss',
                                          ],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/asr_acc', 'validation/main/asr_acc'],
                                         'epoch', file_name='acc.png'))

    # Save best models
    def torch_save(path, _):
        if ngpu > 1:
            torch.save(model.module.state_dict(), path)
            torch.save(model.module, path + ".pkl")
        else:
            torch.save(model.state_dict(), path)
            torch.save(model, path + ".pkl")

    trainer.extend(extensions.snapshot_object(model, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))

    # epsilon decay in the optimizer
    def torch_load(path, obj):
        if ngpu > 1:
            model.module.load_state_dict(torch.load(path))
        else:
            model.load_state_dict(torch.load(path))
        return obj
    if args.opt == 'adadelta':
        if args.criterion == 'acc' and mtl_mode is not 'ctc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/asr_acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/asr_acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(50, 'iteration')))
    report_keys = ['epoch', 'iteration', 'elapsed_time',
                   'main/loss', 'main/asr_loss', 'main/tts_loss', 'main/s2s_loss', 'main/t2t_loss',
                   'main/asr_acc', 'main/t2t_acc',
                   'validation/main/loss', 'validation/main/asr_loss', 'validation/main/tts_loss',
                   'validation/main/s2s_loss', 'validation/main/t2t_loss',
                   'validation/main/asr_acc', 'validation/main/t2t_acc']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(50, 'iteration'))
        report_keys.append('eps')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(50, 'iteration'))

    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()


def recog(args):
    '''Run recognition'''
    # seed setting
    torch.manual_seed(args.seed)

    # read training config
    with open(args.model_conf, "rb") as f:
        logging.info('reading a model config file from' + args.model_conf)
        idim, odim, train_args = pickle.load(f)

    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    logging.info('reading model parameters from' + args.model)
    e2e = E2E(idim, odim, train_args)
    model = Loss(e2e, train_args.mtlalpha)

    def cpu_loader(storage, location):
        return storage

    def remove_dataparallel(state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        return new_state_dict

    model.load_state_dict(remove_dataparallel(torch.load(args.model, map_location=cpu_loader)))

    # read rnnlm
    if args.rnnlm:
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(len(train_args.char_list), 650))
        rnnlm.load_state_dict(torch.load(args.rnnlm, map_location=cpu_loader))
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        if not args.word_dict:
            logging.error('word dictionary file is not specified for the word RNNLM.')
            sys.exit(1)

        word_dict = load_labeldict(args.word_dict)
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(len(word_dict), 650))
        word_rnnlm.load_state_dict(torch.load(args.word_rnnlm, map_location=cpu_loader))
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(word_rnnlm.predictor,
                                           rnnlm.predictor, word_dict, char_dict))
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor,
                                              word_dict, char_dict))

    # read json data
    with open(args.recog_json, 'rb') as f:
        recog_json = json.load(f)['utts']

    new_json = {}
    for name in recog_json.keys():
        feat = kaldi_io_py.read_mat(recog_json[name]['input'][0]['feat'])
        nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm=rnnlm)
        # get 1best and remove sos
        y_hat = nbest_hyps[0]['yseq'][1:]
        y_true = map(int, recog_json[name]['output'][0]['tokenid'].split())

        # print out decoding result
        seq_hat = [train_args.char_list[int(idx)] for idx in y_hat]
        seq_true = [train_args.char_list[int(idx)] for idx in y_true]
        seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
        seq_true_text = "".join(seq_true).replace('<space>', ' ')
        logging.info("groundtruth[%s]: " + seq_true_text, name)
        logging.info("prediction [%s]: " + seq_hat_text, name)

        # copy old json info
        new_json[name] = dict()
        new_json[name]['utt2spk'] = recog_json[name]['utt2spk']

        # added recognition results to json
        logging.debug("dump token id")
        out_dic = dict()
        for _key in recog_json[name]['output'][0]:
            out_dic[_key] = recog_json[name]['output'][0][_key]

        # TODO(karita) make consistent to chainer as idx[0] not idx
        out_dic['rec_tokenid'] = " ".join([str(idx) for idx in y_hat])
        logging.debug("dump token")
        out_dic['rec_token'] = " ".join(seq_hat)
        logging.debug("dump text")
        out_dic['rec_text'] = seq_hat_text

        new_json[name]['output'] = [out_dic]
        # TODO(nelson): Modify this part when saving more than 1 hyp is enabled
        # add n-best recognition results with scores
        if args.beam_size > 1 and len(nbest_hyps) > 1:
            for i, hyp in enumerate(nbest_hyps):
                y_hat = hyp['yseq'][1:]
                seq_hat = [train_args.char_list[int(idx)] for idx in y_hat]
                seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
                new_json[name]['rec_tokenid' + '[' + '{:05d}'.format(i) + ']'] = " ".join([str(idx) for idx in y_hat])
                new_json[name]['rec_token' + '[' + '{:05d}'.format(i) + ']'] = " ".join(seq_hat)
                new_json[name]['rec_text' + '[' + '{:05d}'.format(i) + ']'] = seq_hat_text
                new_json[name]['score' + '[' + '{:05d}'.format(i) + ']'] = hyp['score']

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_json}, indent=4, sort_keys=True).encode('utf_8'))
