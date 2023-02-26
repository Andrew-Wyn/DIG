import os, sys, numpy as np, pickle, argparse
from sklearn.neighbors import kneighbors_graph

import torch


def main(args):
	device = torch.device("cpu")

	if args.model == "bert":
		from bert_helper import nn_init, get_word_embeddings
	else:
		from roberta_helper import nn_init, get_word_embeddings

	print(f'Starting KNN computation...')

	_, tokenizer		= nn_init(device, args.modelname, returns=True)
	word_features		= get_word_embeddings().cpu().detach().numpy()
	word_idx_map		= tokenizer.get_vocab()
	A					= kneighbors_graph(word_features, args.nbrs, mode='distance', n_jobs=args.procs)

	knn_fname = f'processed/knns/{args.model}_{args.nbrs}.pkl'
	with open(knn_fname, 'wb') as f:
		pickle.dump([word_idx_map, word_features, A], f)

	print(f'Written KNN data at {knn_fname}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='knn')
	parser.add_argument('-modelname', type=str)
	parser.add_argument('-model', type=str)
	parser.add_argument('-procs',	default=12, type=int)
	parser.add_argument('-nbrs',  	default=500, type=int)

	args = parser.parse_args()

	main(args)
