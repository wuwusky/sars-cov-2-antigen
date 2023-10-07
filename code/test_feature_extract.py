import logging
import os
import sys

import torch
import omegafold as of
import numpy as np

# from omegafold import pipeline
import argparse
from torch.utils.hipify import hipify_python
from omegafold.pipeline import _set_precision, _get_device, seq2input, get_seq_fasta
from tqdm import tqdm

def get_args_test():
    """
    Parse the arguments, which includes loading the weights

    Returns:
        input_file: the path to the FASTA file to load sequences from.
        output_dir: the output folder directory in which the PDBs will reside.
        batch_size: the batch_size of each forward
        weights: the state dict of the model

    """
    parser = argparse.ArgumentParser(
        description=
        """
        Launch OmegaFold and perform inference on the data. 
        Some examples (both the input and output files) are included in the 
        Examples folder, where each folder contains the output of each 
        available model from model1 to model3. All of the results are obtained 
        by issuing the general command with only model number chosen (1-3).
        """
    )
    # parser.add_argument(
    # )
    parser.add_argument(
        '--num_cycle', default=1, type=int,
        help="The number of cycles for optimization, default to 10"
    )
    parser.add_argument(
        '--subbatch_size', default=128, type=int,
        help=
        """
        The subbatching number, 
        the smaller, the slower, the less GRAM requirements. 
        Default is the entire length of the sequence.
        This one takes priority over the automatically determined one for 
        the sequences
        """
    )
    parser.add_argument(
        '--device', default=None, type=str,
        help=
        'The device on which the model will be running, '
        'default to the accelerator that we can find'
    )
    parser.add_argument(
        '--pseudo_msa_mask_rate', default=0.12, type=float,
        help='The masking rate for generating pseudo MSAs'
    )
    parser.add_argument(
        '--num_pseudo_msa', default=0, type=int,
        help='The number of pseudo MSAs'
    )
    parser.add_argument(
        '--allow_tf32', default=True, type=hipify_python.str2bool,
        help='if allow tf32 for speed if available, default to True'
    )

    args = parser.parse_args()
    _set_precision(args.allow_tf32)

    # weights_url = args.weights
    # weights_file = args.weights_file
    # # if the output directory is not provided, we will create one alongside the
    # # input fasta file
    # if weights_file or weights_url:
    #     weights = _load_weights(weights_url, weights_file)
    #     weights = weights.pop('model', weights)
    # else:
    #     weights = None
    try:
        weights = torch.load('./user_data/model_data_s3/model.pt', map_location='cpu')
    except Exception as e:
        print(e)
        weights = torch.load('e:/model.pt', map_location='cpu')
        pass
    weights = weights.pop('model', weights)

    forward_config = argparse.Namespace(
        subbatch_size=args.subbatch_size,
        num_recycle=args.num_cycle,
    )

    args.device = _get_device(args.device)

    return args, weights, forward_config

args, state_dict, forward_config = get_args_test()

@torch.no_grad()
def data_process_sim(input_seq, max_len = 1024):
    input_data = seq2input(
        input_seq,
        num_pseudo_msa=args.num_pseudo_msa,
        device=args.device,
        mask_rate=args.pseudo_msa_mask_rate,
        num_cycle=args.num_cycle,
        max_len=max_len,
    )
    return input_data


if __name__ == '__main__':
    input_dir = './fasta_antigen/'
    list_files = os.listdir(input_dir)[:]
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    args, state_dict, forward_config = get_args_test()
    logging.info(f'constructing omegafold')
    model = of.OmegaFold(of.make_config())
    if "model" in state_dict:
        state_dict = state_dict.pop("model")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)

    for temp_file in tqdm(list_files, ncols=100):
    # for temp_file in list_files:
        temp_input_dir = input_dir + temp_file

        # input_data = data_process(temp_input_dir)

        temp_seqs = get_seq_fasta(temp_input_dir)

        for temp_seq in temp_seqs.values():
            break
        input_data = data_process_sim(temp_seq)
        with torch.no_grad():
            features = model.feature_extract(input_data, forward_config)
            f_node, f_edge = features
            print('')
            print(f_node.shape)
            print(f_edge.shape)
            f_node_np = f_node.cpu().numpy()
            f_edge_np = f_edge.cpu().numpy()

            temp_file_name = temp_file.split('.')[0]
            temp_save_dir = './feats_antigen/' + temp_file_name + '_node.npy'
            np.save(temp_save_dir, f_node_np, allow_pickle=True)
            temp_save_dir = './feats_antigen/' + temp_file_name + '_edge.npy'
            np.save(temp_save_dir, f_edge_np, allow_pickle=True)
            

            # features = model.feature_extract_cycle(input_data, forward_config)
    print('test ok')

