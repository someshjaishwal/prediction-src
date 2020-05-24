'''

CUDA_VISIBLE_DEVICES=2 python evaluate.py --cuda --model-path models/network.pth --manifest data.csv 
'''


import sys
sys.path.insert(0,'./files')

import warnings
warnings.simplefilter('ignore')

import argparse
import numpy as np
import torch
from tqdm import tqdm

from data_loader import SpectrogramDataset2, AudioDataLoader
from decoder import GreedyDecoder
from opts import add_decoder_args, add_inference_args
from utils import load_model, rectify

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--manifest', metavar='DIR',
                    help='path to validation manifest csv', default='manifest.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for testing')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")
parser = add_decoder_args(parser)

def evaluate(test_loader, device, model, decoder, target_decoder, save_output=False, verbose=False, half=False):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    correct, nb_items = 0, 0
    output_data = []
    for i, (data) in enumerate(test_loader): #tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        if half:
            inputs = inputs.half()
            
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out, output_sizes = model(inputs, input_sizes)

        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)

        if save_output is not None:
            # add output to data array, and continue
            output_data.append((out.cpu().numpy(), output_sizes.numpy(), target_strings))
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            
            ## edit distance based wer cer
            both_transcripts = rectify(transcript, decoder)
            transcriptEdit = both_transcripts['cer']
            
            if transcriptEdit != reference:
                verbose = True
            else:
                correct = correct + 1 #(transcriptEdit == reference)
                verbose = False
                
            nb_items = nb_items + 1
            if verbose:
                print("Original :", reference.lower())
                print("Predicted :", transcriptEdit.lower())
                print()
                
    return 100 * correct/nb_items

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.half)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    else:
        decoder = None

    target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))

    '''
        dataset - 
                v
    '''
    test_dataset = SpectrogramDataset2(audio_conf=model.audio_conf, manifest_filepath=args.manifest,
                                      labels=model.labels, normalize=True)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    acc = evaluate(test_loader=test_loader,
                                     device=device,
                                     model=model,
                                     decoder=decoder,
                                     target_decoder=target_decoder,
                                     save_output=args.save_output,
                                     verbose=args.verbose,
                                     half=args.half)

    print(f'\n\nSummary : Top-1 Accuaracy = {acc:.3f}%')

    if args.save_output is not None:
        np.save(args.save_output, output_data)

