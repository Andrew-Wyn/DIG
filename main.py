import os
import json
import csv
import sys, numpy as np, argparse, random
sys.path.append('../')
from random import shuffle
from collections import defaultdict
import csv

from tqdm import tqdm

import pandas as pd

import torch
from datasets import load_dataset
from datasets import Dataset
from dig import DiscretetizedIntegratedGradients
from attributions import run_dig_explanation
from metrics import eval_log_odds, eval_anti_log_odds, regression_eval_log_odds, regression_eval_anti_log_odds
import monotonic_paths

import os, argparse
from sklearn.neighbors import kneighbors_graph

import torch

from roberta_helper import nn_init, get_word_embeddings


# ROBERTA | XLM_ROBERTA | GILBERTO(CAMEMBERT) are all roberta implementation
from roberta_helper import nn_init, load_mappings, nn_forward_func, predict, get_mask_token_emb, get_inputs, get_tokens

all_outputs = []

# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, ff, get_tokens):
	# computes the attributions for given input

	# move inputs to main device
	inp = [x.to(device) if x is not None else None for x in inputs]
	scaled_features, _, _, input_embed, _, position_embed, _, type_embed, _, attention_mask = inp

	# compute attribution
	attr = run_dig_explanation(attr_func, scaled_features, position_embed, type_embed, attention_mask, (2**args.factor)*(args.steps+1)+1)

	# compute metrics
	log_odd, lo_p, lo_p_p		= eval_log_odds(ff, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk_1)
	anti_log_odd, alo_p, alo_p_p = eval_anti_log_odds(ff, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk_2)

	return (log_odd, lo_p, lo_p_p), (anti_log_odd, alo_p, alo_p_p)


def regression_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, ff, get_tokens):
	# computes the attributions for given input

	# move inputs to main device
	inp = [x.to(device) if x is not None else None for x in inputs]
	scaled_features, _, _, input_embed, _, position_embed, _, type_embed, _, attention_mask = inp

	# compute attribution
	attr = run_dig_explanation(attr_func, scaled_features, position_embed, type_embed, attention_mask, (2**args.factor)*(args.steps+1)+1)

	# compute metrics
	log_odd, lo_p, lo_p_p			= regression_eval_log_odds(ff, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk_1)
	anti_log_odd, alo_p, alo_p_p 	= regression_eval_anti_log_odds(ff, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk_2)

	return (log_odd, lo_p, lo_p_p), (anti_log_odd, alo_p, alo_p_p)


def read_complexity_dataset(path=None):
    data = []

    df = pd.read_csv(path)

    for _, row in df.iterrows():
        
        num_individuals = 20

        text = row["SENTENCE"]

        label = 0

        for i in range(num_individuals):
            label += int(row[f"judgement{i+1}"])

        label = label/num_individuals

        data.append({
            "text": text,
            "label": label
        })


    return Dataset.from_list(data)


def create_dataset_from_faulty_csv(src_path):
    dataset_dict = {'text': [], 'label_pos': [], 'label_neg': []}
    with open(src_path) as src_file:
        csv_reader = csv.reader(src_file, delimiter=',', quotechar='"')
        print('')
        for row in csv_reader:
            if row[0] == 'idtwitter':
                continue
            if len(row) != 9:
                cut_row = row[:9]
                cut_row[8] += ',' + ', '.join(row[9:])
                row = cut_row
            dataset_dict['text'].append(row[8])
            dataset_dict['label_pos'].append(int(row[2]))
            dataset_dict['label_neg'].append(int(row[3]))
    return Dataset.from_dict(dataset_dict)


def sst2_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics):
	# Define the Attribution function
	def ff(input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False):
			return nn_forward_func(predict, input_embed, attention_mask, position_embed, type_embed, return_all_logits)
	attr_func = DiscretetizedIntegratedGradients(ff)

	(log_odd, lo_p, lo_p_p), (anti_log_odd, alo_p, alo_p_p) = classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, ff, get_tokens)
	
	xai_metrics["log_odd"] += log_odd
	xai_metrics["anti_log_odd"] += anti_log_odd

	return (log_odd, lo_p, lo_p_p), (anti_log_odd, alo_p, alo_p_p)


def complexity_binary_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics):
	# Define the Attribution function
	def ff(input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False):
			return nn_forward_func(predict, input_embed, attention_mask, position_embed, type_embed, return_all_logits)
	attr_func = DiscretetizedIntegratedGradients(ff)

	
	(log_odd, lo_p, lo_p_p), (anti_log_odd, alo_p, alo_p_p) = classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, ff, get_tokens)
		
	xai_metrics["log_odd"] += log_odd
	xai_metrics["anti_log_odd"] += anti_log_odd

	return (log_odd, lo_p, lo_p_p), (anti_log_odd, alo_p, alo_p_p)


def complexity_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics):
	# Define the Attribution function
	def ff(input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False):
			return nn_forward_func(predict, input_embed, attention_mask, position_embed, type_embed, return_all_logits)
	attr_func = DiscretetizedIntegratedGradients(ff)

	(log_odd, lo_p, lo_p_p), (anti_log_odd, alo_p, alo_p_p) = regression_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, ff, get_tokens)
	
	xai_metrics["reg_log_odd"] += log_odd
	xai_metrics["reg_anti_log_odd"] += anti_log_odd

	return (log_odd, lo_p, lo_p_p), (anti_log_odd, alo_p, alo_p_p)

	
def sentpolc_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics):
		
	# Define the predict function for pos sentipolc task
	def pos_predict(model, inputs_embeds, attention_mask=None):
		return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)["logits"]["pos"]

	# Define the Attribution function
	def ff(input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False):
			return nn_forward_func(pos_predict, input_embed, attention_mask, position_embed, type_embed, return_all_logits)
	attr_func = DiscretetizedIntegratedGradients(ff)

	(log_odd_pos, lo_p_pos, lo_p_p_pos), (anti_log_odd_pos, alo_p_pos, alo_p_p_pos) = classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, ff, get_tokens)

	# Define the predict function for neg sentipolc task
	def neg_predict(model, inputs_embeds, attention_mask=None):
		return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)["logits"]["neg"]
	
	# Define the Attribution function
	def ff(input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False):
			return nn_forward_func(neg_predict, input_embed, attention_mask, position_embed, type_embed, return_all_logits)
	attr_func = DiscretetizedIntegratedGradients(ff)

	(log_odd_neg, lo_p_neg, lo_p_p_neg), (anti_log_odd_neg, alo_p_neg, alo_p_p_neg) = classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, ff, get_tokens)
	
	xai_metrics["log_odd_pos"] += log_odd_pos
	xai_metrics["anti_log_odd_pos"] += anti_log_odd_pos
	xai_metrics["log_odd_neg"] += log_odd_neg
	xai_metrics["anti_log_odd_neg"] += anti_log_odd_neg

	return (log_odd_pos, lo_p_pos, lo_p_p_pos), (anti_log_odd_pos, alo_p_pos, alo_p_p_pos), (log_odd_neg, lo_p_neg, lo_p_p_neg), (anti_log_odd_neg, alo_p_neg, alo_p_p_neg)


def average_mertrics(metrics, iterations):
	averaged_metrics = {}

	for k, v in metrics.items():
		averaged_metrics[k] = v/iterations

	return averaged_metrics


def knn_main(args):
	"""
		Compute the KNN graph for the tokens embedding space
	"""

	device = torch.device("cpu")

	print('=========== KNN Computation ===========')

	# Initiliaze the tokenizer
	_, tokenizer		= nn_init(device, args.modelname, args.task, returns=True)

	word_features		= get_word_embeddings().cpu().detach().numpy()
	word_idx_map		= tokenizer.get_vocab()
	# Compute the (weighted) graph of k-Neighbors for points in word_features -> the single token's ids.
	adj					= kneighbors_graph(word_features, args.knn_nbrs, mode='distance', n_jobs=args.procs)

	print("=========== DONE! ===========")

	return word_idx_map, word_features, adj


def main(args):
	# set seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Load the token's embeddings KNN graph
	auxiliary_data = knn_main(args)

	# Fix the gpu to use
	device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# init model and tokenizer in cpu first
	nn_init(device, args.modelname, args.task)

	if args.dataset is None:
		# load the dataset
		dataset	= load_dataset('sst2')['test']
		data	= list(zip(dataset['sentence'], dataset['label'], dataset['idx']))
	else:
		if args.task == "complexity" or args.task == "complexity_binary":
			dataset = read_complexity_dataset(args.dataset)
			data = list(zip(dataset["text"],))
		elif args.task == "sentipolc": # TODO: add italian sentiment analysis loading
			dataset = create_dataset_from_faulty_csv(args.dataset)
			data = list(zip(dataset["text"],))

	# shuffle the data 
	shuffle(data)

	# get ref token embedding
	# DIG MODIFICATION: usage of MASK token instead of PAD as base.
	mask_token_emb = get_mask_token_emb(device)

	# compute the DIG attributions for all the inputs
	print('Starting attribution computation...')
	inputs = []

	# dictionary where the metrics will be saved
	xai_metrics = defaultdict(lambda: 0)

	# iteration stuffs
	count=0
	print_step = 10
	max_iterations = 100

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	with open(f"{args.output_dir}/sentences.csv", 'w') as csv_file:
		writer = csv.writer(csv_file, delimiter=',') # init the csv writer object

		# write header
		if args.task == "sentipolc":
			writer.writerow(["sentence", "pos log_odd", "pos log_odd prob", "pos log_odd prob perturbed", "pos anti log_odd", "pos anti log_odd prob", "pos anti log_odd prob perturbed", "neg log_odd", "neg log_odd prob", "neg log_odd prob perturbed", "neg anti log_odd", "neg anti log_odd prob", "neg anti log_odd prob perturbed"])
		else:
			writer.writerow(["sentence", "log_odd", "log_odd prob", "log_odd prob perturbed", "anti log_odd", "anti log_odd prob", "anti log_odd prob perturbed"])

		for row in tqdm(data[:max_iterations]):
			# augment the input with contour informations needed by DIG attribution score
			inp = get_inputs(row[0], device)
			input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp

			# generates the paths required by DIG
			scaled_features 		= monotonic_paths.scale_inputs(input_ids.squeeze().tolist(), ref_input_ids.squeeze().tolist(),\
												device, auxiliary_data, steps=args.steps, factor=args.factor, strategy=args.strategy)
			
			inputs					= [scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, \
												ref_position_embed, type_embed, ref_type_embed, attention_mask]

			if args.task == "complexity": # call regression metrics
				(log_odd, lo_p, lo_p_p), (anti_log_odd, alo_p, alo_p_p) = complexity_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics)
				# write the sentence's metrics and probs over the csv file
				writer.writerow([row[0], str(log_odd), str(lo_p), str(lo_p_p), str(anti_log_odd), str(alo_p), str(alo_p_p)])
			elif args.task == "complexity_binary": # call classification metrics
				(log_odd, lo_p, lo_p_p), (anti_log_odd, alo_p, alo_p_p) = complexity_binary_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics)
				# write the sentence's metrics and probs over the csv file
				writer.writerow([row[0], str(log_odd), str(lo_p), str(lo_p_p), str(anti_log_odd), str(alo_p), str(alo_p_p)])
			elif args.task == "sst2": # call classification metrics
				(log_odd, lo_p, lo_p_p), (anti_log_odd, alo_p, alo_p_p) = sst2_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics)
				# write the sentence's metrics and probs over the csv file
				writer.writerow([row[0], str(log_odd), str(lo_p), str(lo_p_p), str(anti_log_odd), str(alo_p), str(alo_p_p)])
			elif args.task == "sentipolc": # call classification metrics twice one for each sub-task
				(log_odd_pos, lo_p_pos, lo_p_p_pos), (anti_log_odd_pos, alo_p_pos, alo_p_p_pos), (log_odd_neg, lo_p_neg, lo_p_p_neg), (anti_log_odd_neg, alo_p_neg, alo_p_p_neg) = sentpolc_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics)
				# write the sentence's metrics and probs over the csv file
				writer.writerow([row[0], str(log_odd_pos), str(lo_p_pos), str(lo_p_p_pos), str(anti_log_odd_pos), str(alo_p_pos), str(alo_p_p_pos), str(log_odd_neg), str(lo_p_neg), str(lo_p_p_neg), str(anti_log_odd_neg), str(alo_p_neg), str(alo_p_p_neg)])

			count += 1

			# print the metrics
			if count % print_step == 0:
				print(average_mertrics(xai_metrics, count))

	print(average_mertrics(xai_metrics, count))

	with open(f"{args.output_dir}/xai_metrics.json", 'w') as f:
		json.dump(average_mertrics(xai_metrics, count), f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='IG Path')
	parser.add_argument('-modelname', 	default="xlm-roberta-base", type=str) # path or hf ref of the used model
	parser.add_argument('-modeltype', 	default="roberta", type=str) # type of the model: roberta | camem| xlm
	parser.add_argument('-task', 		default="complexity", type=str) # taks over which perform DIG computation: complexity | sst2 | sentiment_it
	parser.add_argument('-runtype', 	default="hf_np", type=str) # type of model over which compute the DIG
	parser.add_argument('-dataset', 	default=None, type=str) # dataset used, if None SST2 will be used
	parser.add_argument('-strategy', 	default='greedy', choices=['greedy', 'maxcount'], help='The algorithm to find the next anchor point')
	parser.add_argument('-steps', 		default=30, type=int)	# m
	parser.add_argument('-topk_1', 		default=10, type=int)	# k
	parser.add_argument('-topk_2', 		default=50, type=int)	# k
	parser.add_argument('-factor', 		default=0, 	type=int)	# f
	parser.add_argument('-knn_nbrs',	default=200, type=int)	# KNN
	parser.add_argument('-seed', 		default=42, type=int)
	parser.add_argument('-procs',	default=5, type=int)
	parser.add_argument('-output_dir', 	type=str)

	args = parser.parse_args()

	main(args)
