import os
import random
import math
import torch
import argparse
from argparse import Namespace
from utils.args_utils import str2list, str2bool
import copy
from time import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from data.mscoco2014_dataset_manager import MsCocoDataLoader, MsCocoDatasetKarpathy
from utils import language_utils
from utils.language_utils import compute_num_pads
from utils.parallel_utils import dist_gather_object

from models.transformer import Transformer

from eval.eval import COCOEvalCap


import functools
print = functools.partial(print, flush=True)


def compute_evaluation_phase_loss(loss_function,
                                  model,
                                  data_set,
                                  data_loader,
                                  num_samples,
                                  sub_batch_size,
                                  dataset_split,
                                  rank=0,
                                  verbose=False):
    model.eval()

    sb_size = sub_batch_size
    batch_input_x, batch_target_y, batch_input_x_num_pads, batch_target_y_num_pads, batch_rand_indexes \
        = data_loader.get_random_samples(num_samples=num_samples,
                                         dataset_split=dataset_split)

    tot_loss = 0
    num_sub_batch = int(num_samples / sb_size)
    tot_num_tokens = 0
    for sb_it in range(num_sub_batch):
        from_idx = sb_it * sb_size
        to_idx = (sb_it + 1) * sb_size

        sub_batch_input_x = batch_input_x[from_idx: to_idx].to(rank)
        sub_batch_target_y = batch_target_y[from_idx: to_idx].to(rank)
        tot_num_tokens += sub_batch_target_y.size(1)*sub_batch_target_y.size(0) - \
                          sum(batch_target_y_num_pads[from_idx: to_idx])
        pred = model(enc_x=sub_batch_input_x,
                     dec_x=sub_batch_target_y[:, :-1],
                     enc_x_num_pads=batch_input_x_num_pads[from_idx: to_idx],
                     dec_x_num_pads=batch_target_y_num_pads[from_idx: to_idx],
                     apply_softmax=False)
        tot_loss += loss_function(pred, sub_batch_target_y[:, 1:],
                                  data_set.get_pad_token_idx(),
                                  divide_by_non_zeros=False).item()
    tot_loss /= tot_num_tokens
    if verbose and rank == 0:
        print("Validation Loss on " + str(num_samples) + " samples: " + str(tot_loss))

    return tot_loss


def parallel_evaluate_model(ddp_model,
                            validate_x, validate_y,
                            y_idx2word_list,
                            beam_size, max_seq_len,
                            sos_idx, eos_idx,
                            rank, world_size, ddp_sync_port,
                            parallel_batches=16, verbose=True,
                            return_cider=False,
                            stanford_model_path="./eval/get_stanford_models.sh"):
    # avoid synchronization problems in case of ugly numbers
    assert (len(validate_x) % world_size == 0), "to ensure correctness num test sentences must be multiple of world size" \
        + " for the sake of a cleaner code, maybe in future implementation will be allowed."
    start_time = time()

    sub_list_predictions = []

    # divide validate_x and validate_y by the number of gpus
    sub_sample_size = math.ceil(len(validate_x) / float(world_size))
    sub_validate_x = validate_x[sub_sample_size * rank: sub_sample_size * (rank + 1)]

    ddp_model.eval()
    with torch.no_grad():
        sb_size = parallel_batches
        num_iter_sub_batches = math.ceil(len(sub_validate_x) / sb_size)
        for sb_it in range(num_iter_sub_batches):
            last_iter = sb_it == num_iter_sub_batches - 1
            if last_iter:
                from_idx = sb_it * sb_size
                to_idx = len(sub_validate_x)
            else:
                from_idx = sb_it * sb_size
                to_idx = (sb_it + 1) * sb_size

            sub_batch_x = torch.nn.utils.rnn.pad_sequence(sub_validate_x[from_idx: to_idx], batch_first=True).to(rank)
            sub_batch_x_num_pads = compute_num_pads(sub_validate_x[from_idx: to_idx])

            beam_search_kwargs = {'beam_size': beam_size,
                                  'beam_max_seq_len': max_seq_len,
                                  'sample_or_max': 'max',
                                  'how_many_outputs': 1,
                                  'sos_idx': sos_idx,
                                  'eos_idx': eos_idx}

            output_words, _ = ddp_model(enc_x=sub_batch_x,
                                        enc_x_num_pads=sub_batch_x_num_pads,
                                        mode='beam_search', **beam_search_kwargs)

            # take the first element of beam_size sequences
            output_words = [output_words[i][0] for i in range(len(output_words))]

            pred_sentence = language_utils.convert_allsentences_idx2word(output_words, y_idx2word_list)
            for sentence in pred_sentence:
                sub_list_predictions.append(' '.join(sentence[1:-1]))  # remove EOS and SOS

    ddp_model.train()

    dist.barrier()

    list_sub_predictions = dist_gather_object(sub_list_predictions,
                                              rank,
                                              dst_rank=0,
                                              sync_port=ddp_sync_port)

    if rank == 0 and verbose:
        list_predictions = []
        for sub_predictions in list_sub_predictions:
            list_predictions += sub_predictions

        list_list_references = []
        for i in range(len(validate_x)):
            target_references = []
            for j in range(len(validate_y[i])):
                target_references.append(validate_y[i][j])
            list_list_references.append(target_references)

        gts_dict = dict()
        for i in range(len(list_list_references)):
            gts_dict[i] = [{u'image_id': i, u'caption': list_list_references[i][j]}
                           for j in range(len(list_list_references[i]))]

        pred_dict = dict()
        for i in range(len(list_predictions)):
            pred_dict[i] = [{u'image_id': i, u'caption': list_predictions[i]}]

        coco_eval = COCOEvalCap(gts_dict, pred_dict, list(range(len(list_predictions))),
                                get_stanford_models_path=stanford_model_path)
        score_results = coco_eval.evaluate(bleu=True, rouge=True, cider=True, spice=True, meteor=True, verbose=False)
        elapsed_ticks = time() - start_time
        print("Evaluation Phase over " + str(len(validate_x)) + " BeamSize: " + str(beam_size) +
              "  elapsed: " + str(int(elapsed_ticks/60)) + " m " + str(int(elapsed_ticks % 60)) + ' s')
        print(score_results)

    dist.barrier()

    if return_cider:
        cider = score_results[0]
        _, cider = cider
    else:
        cider = None

    if rank == 0:
        return pred_dict, gts_dict, cider

    return None, None, cider


def evaluate_model_on_set(ddp_model,
                          caption_idx2word_list,
                          sos_idx, eos_idx,
                          num_samples,
                          data_loader,
                          dataset_split,
                          eval_max_len,
                          rank, world_size, ddp_sync_port,
                          parallel_batches=16,
                          beam_sizes=[1],
                          stanford_model_path='./eval/get_stanford_models.sh',
                          get_predictions=False):

    with torch.no_grad():
        ddp_model.eval()
        indexes = range(num_samples)
        val_x = [data_loader.get_bboxes_by_idx(i, dataset_split=dataset_split)
                 for i in indexes]
        val_y = [data_loader.get_all_image_captions_by_idx(i, dataset_split=dataset_split)
                 for i in indexes]
        for beam in beam_sizes:
            pred_dict, gts_dict, cider = parallel_evaluate_model(ddp_model, val_x, val_y,
                                                                 y_idx2word_list=caption_idx2word_list,
                                                                 beam_size=beam, max_seq_len=eval_max_len,
                                                                 sos_idx=sos_idx, eos_idx=eos_idx,
                                                                 rank=rank, world_size=world_size,
                                                                 ddp_sync_port=ddp_sync_port,
                                                                 parallel_batches=parallel_batches,
                                                                 verbose=True,
                                                                 return_cider=True,
                                                                 stanford_model_path=stanford_model_path)

            if rank == 0 and get_predictions:
                return pred_dict, gts_dict, cider

    return None, None, cider


def eval_on_cider_only(ddp_model,
                       y_idx2word_list,
                       sos_idx, eos_idx,
                       num_samples,
                       data_loader,
                       dataset_split,
                       max_seq_len,
                       rank,
                       beam_size,
                       parallel_batches=16,
                       stanford_model_path='./eval/get_stanford_models.sh'):

    indexes = range(num_samples)
    validate_x = [data_loader.get_bboxes_by_idx(i, dataset_split=dataset_split)
             for i in indexes]
    validate_y = [data_loader.get_all_image_captions_by_idx(i, dataset_split=dataset_split)
             for i in indexes]

    ddp_model.eval()

    list_predictions = []
    list_list_references = []

    sb_size = parallel_batches
    num_iter_sub_batches = math.ceil(len(validate_x) / sb_size)
    for sb_it in range(num_iter_sub_batches):
        last_iter = sb_it == num_iter_sub_batches - 1
        if last_iter:
            from_idx = sb_it * sb_size
            to_idx = len(validate_x)
        else:
            from_idx = sb_it * sb_size
            to_idx = (sb_it + 1) * sb_size
        sub_batch_x = torch.nn.utils.rnn.pad_sequence(validate_x[from_idx: to_idx], batch_first=True).to(rank)
        sub_batch_x_num_pads = compute_num_pads(validate_x[from_idx: to_idx])

        beam_search_kwargs = {'beam_size': beam_size,
                              'beam_max_seq_len': max_seq_len,
                              'sample_or_max': 'max',
                              'how_many_outputs': 1,
                              'sos_idx': sos_idx,
                              'eos_idx': eos_idx}

        output_words, _ = ddp_model(enc_x=sub_batch_x,
                                    enc_x_num_pads=sub_batch_x_num_pads,
                                    mode='beam_search', **beam_search_kwargs)

        del sub_batch_x

        # take the first element of beam_size sequences
        output_words = [output_words[i][0] for i in range(len(output_words))]

        pred_sentence = language_utils.convert_allsentences_idx2word(output_words, y_idx2word_list)
        for sentence in pred_sentence:
            list_predictions.append(' '.join(sentence[1:-1]))  # remove EOS and SOS

    for i in range(len(validate_x)):
        target_references = []
        for j in range(len(validate_y[i])):
            target_references.append(validate_y[i][j])
        list_list_references.append(target_references)

    gts_dict = dict()
    for i in range(len(list_list_references)):
        gts_dict[i] = [{u'image_id': i, u'caption': list_list_references[i][j]}
                       for j in range(len(list_list_references[i]))]

    pred_dict = dict()
    for i in range(len(list_predictions)):
        pred_dict[i] = [{u'image_id': i, u'caption': list_predictions[i]}]

    coco_eval = COCOEvalCap(gts_dict, pred_dict, list(range(len(list_predictions))),
                            get_stanford_models_path=stanford_model_path)
    score_results = coco_eval.evaluate(bleu=False, rouge=False, cider=True,
                                       spice=False, meteor=False, verbose=False)

    del validate_x, validate_y

    cider = score_results[0]
    _, cider = cider
    return cider

