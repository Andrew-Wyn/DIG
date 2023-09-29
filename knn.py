import os, sys, numpy as np, pickle, argparse
from sklearn.neighbors import kneighbors_graph

import torch

from roberta_helper import nn_init, get_word_embeddings

# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
	device = torch.device("cpu")

	print(f'Starting KNN computation...')

	_, tokenizer		= nn_init(device, args.modelname, returns=True)
	word_features		= get_word_embeddings().cpu().detach().numpy()
	word_idx_map		= tokenizer.get_vocab()
	A					= kneighbors_graph(word_features, args.nbrs, mode='distance', n_jobs=args.procs)

	# knn_fname = f'processed/knns/{args.finetuningtype}_{args.modeltype}_{args.runname}_{args.nbrs}.pkl'

	knn_fname = 'processed/knns/knn.pkl'
	with open(knn_fname, 'wb') as f:
		pickle.dump([word_idx_map, word_features, A], f)

	print(f'Written KNN data at {knn_fname}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='knn')
	parser.add_argument('-modelname', default="roberta-base", type=str)
	parser.add_argument('-modeltype', default="roberta", type=str)
	parser.add_argument('-finetuningtype', default='comp_en', type=str)
	parser.add_argument('-runname', default='p_nf', type=str) # not-pretrained-not-finetuntuned | not-pretrained-finetuned ...
	parser.add_argument('-procs',	default=12, type=int)
	parser.add_argument('-nbrs',  	default=500, type=int)

	args = parser.parse_args()

	main(args)
