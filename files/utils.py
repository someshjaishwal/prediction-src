import torch
import torch.distributed as dist

from model2 import DeepSpeech
from transcriptions import transcriptions

def reduce_tensor(tensor, world_size, reduce_op_max=False):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.MAX if reduce_op_max is True else dist.reduce_op.SUM)  # Default to sum
    if not reduce_op_max:
        rt /= world_size
    return rt


def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error


def load_model(device, model_path, use_half = True):
    model = DeepSpeech.load_model(model_path)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()
    return model


def rectify(transcript, decoder):
    trans_corpus = transcriptions
    wi, wer, ci, cer = None, None, None, None
    for i in range(len(trans_corpus)):
        wer_inst = decoder.wer(transcript, trans_corpus[i])
        cer_inst = decoder.cer(transcript, trans_corpus[i])
        
        if wi is None or wer_inst < wer:
            wer, wi = wer_inst, i
        if ci is None or cer_inst < cer:
            cer, ci = cer_inst, i
            
    return {'wer' : trans_corpus[wi], 'cer' : trans_corpus[ci]}

    






















