import os
import json
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
from metrics import eval_log_odds, eval_comprehensiveness, eval_sufficiency, eval_anti_log_odds, regression_eval_log_odds, regression_eval_comprehensiveness, regression_eval_anti_log_odds, regression_eval_sufficiency
import monotonic_paths

# ROBERTA | XLM_ROBERTA | GILBERTO(CAMEMBERT) are all roberta implementation
from roberta_helper import nn_init, load_mappings, nn_forward_func, predict, get_mask_token_emb, get_inputs, get_tokens

all_outputs = []

# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, predict, get_tokens):
	# computes the attributions for given input

	# move inputs to main device
	inp = [x.to(device) if x is not None else None for x in inputs]
	scaled_features, _, _, input_embed, _, position_embed, _, type_embed, _, attention_mask = inp

	# compute attribution
	attr = run_dig_explanation(attr_func, scaled_features, position_embed, type_embed, attention_mask, (2**args.factor)*(args.steps+1)+1)

	# compute metrics
	log_odd, _		= eval_log_odds(nn_forward_func, predict, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)
	anti_log_odd, _ = eval_anti_log_odds(nn_forward_func, predict, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)
	comp			= eval_comprehensiveness(nn_forward_func, predict, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)
	suff			= eval_sufficiency(nn_forward_func, predict, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)

	return log_odd, anti_log_odd, comp, suff


def regression_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, predict, get_tokens):
	# computes the attributions for given input

	# move inputs to main device
	inp = [x.to(device) if x is not None else None for x in inputs]
	scaled_features, _, _, input_embed, _, position_embed, _, type_embed, _, attention_mask = inp

	# compute attribution
	attr = run_dig_explanation(attr_func, scaled_features, position_embed, type_embed, attention_mask, (2**args.factor)*(args.steps+1)+1)

	# compute metrics
	log_odd			= regression_eval_log_odds(nn_forward_func, predict, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)
	anti_log_odd 	= regression_eval_anti_log_odds(nn_forward_func, predict, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)
	comp			= regression_eval_comprehensiveness(nn_forward_func, predict, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)
	suff			= regression_eval_sufficiency(nn_forward_func, predict, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)

	return log_odd, anti_log_odd, comp, suff


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
				return nn_forward_func(predict, input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False)
		attr_func = DiscretetizedIntegratedGradients(ff)

		log_odd, anti_log_odd, comp, suff = classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, predict, get_tokens)
		
		xai_metrics["log_odd"] += log_odd
		xai_metrics["anti_log_odd"] += anti_log_odd
		xai_metrics["comp"] += comp
		xai_metrics["suff"] += suff


def complexity_binary_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics):
	# Define the Attribution function
	def ff(input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False):
			return nn_forward_func(predict, input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False)
	attr_func = DiscretetizedIntegratedGradients(ff)

	
	log_odd, anti_log_odd, comp, suff = classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, predict, get_tokens)
		
	xai_metrics["log_odd"] += log_odd
	xai_metrics["anti_log_odd"] += anti_log_odd
	xai_metrics["comp"] += comp
	xai_metrics["suff"] += suff


def complexity_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics):
		# Define the Attribution function
		def ff(input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False):
				return nn_forward_func(predict, input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False)
		attr_func = DiscretetizedIntegratedGradients(ff)


		log_odd, anti_log_odd, comp, suff = regression_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, predict, get_tokens)
		
		xai_metrics["reg_log_odd"] += log_odd
		xai_metrics["reg_anti_log_odd"] += anti_log_odd
		xai_metrics["reg_comp"] += comp
		xai_metrics["reg_suff"] += suff
	
def sentpolc_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics):
		
		def pos_predict(model, inputs_embeds, attention_mask=None):
			return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)["logits"]["pos"]

		# Define the Attribution function
		def ff(input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False):
				return nn_forward_func(pos_predict, input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False)
		attr_func = DiscretetizedIntegratedGradients(ff)

		log_odd_pos, anti_log_odd_pos, comp_pos, suff_pos = classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, pos_predict, get_tokens)

		def neg_predict(model, inputs_embeds, attention_mask=None):
			return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)["logits"]["neg"]
		
		# Define the Attribution function
		def ff(input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False):
				return nn_forward_func(neg_predict, input_embed, attention_mask=None, position_embed=None, type_embed=None, return_all_logits=False)
		attr_func = DiscretetizedIntegratedGradients(ff)

		log_odd_neg, anti_log_odd_neg, comp_neg, suff_neg = classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, neg_predict, get_tokens)
		
		xai_metrics["log_odd_pos"] += log_odd_pos
		xai_metrics["anti_log_odd_pos"] += anti_log_odd_pos
		xai_metrics["comp_pos"] += comp_pos
		xai_metrics["suff_pos"] += suff_pos
		xai_metrics["log_odd_neg"] += log_odd_neg
		xai_metrics["anti_log_odd_neg"] += anti_log_odd_neg
		xai_metrics["comp_neg"] += comp_neg
		xai_metrics["suff_neg"] += suff_neg


def average_mertrics(metrics, iterations):
	averaged_metrics = {}

	for k, v in metrics.items():
		averaged_metrics[k] = v/iterations

	return averaged_metrics


def main(args):
	# set seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Load the token's embeddings KNN graph
	auxiliary_data = load_mappings(args.task, args.modeltype, args.runtype, knn_nbrs=args.knn_nbrs)

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
	max_iterations = 200

	for i, row in tqdm(enumerate(data[:max_iterations])):
		# augment the input with contour informations needed by DIG attribution score
		inp = get_inputs(row[0], device)
		input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp

		# generates the paths required by DIG
		scaled_features 		= monotonic_paths.scale_inputs(input_ids.squeeze().tolist(), ref_input_ids.squeeze().tolist(),\
											device, auxiliary_data, steps=args.steps, factor=args.factor, strategy=args.strategy)
		
		inputs					= [scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask]

		if args.task == "complexity": # call regression metrics
			complexity_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics)
		elif args.task == "complexity_binary": # call classification metrics
			complexity_binary_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics)
		elif args.task == "sst2": # call classification metrics
			sst2_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics)
		elif args.task == "sentipolc": # call classification metrics twice one for each sub-task
			sentpolc_calculate_attributions(inputs, device, args, mask_token_emb, nn_forward_func, get_tokens, xai_metrics)

		count += 1

		# print the metrics
		if count % print_step == 0:
			print(average_mertrics(xai_metrics, i))

	print(average_mertrics(xai_metrics, max_iterations))

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	with open(f"{args.output_dir}/xai_metrics.json", 'w') as f:
		json.dump(average_mertrics(xai_metrics, max_iterations), f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='IG Path')
	parser.add_argument('-modelname', 	default="xlm-roberta-base", type=str) # path or hf ref of the used model
	parser.add_argument('-modeltype', 	default="roberta", type=str) # type of the model: roberta | camem| xlm
	parser.add_argument('-task', 		default="complexity", type=str) # taks over which perform DIG computation: complexity | sst2 | sentiment_it
	parser.add_argument('-runtype', 	default="hf_np", type=str) # type of model over which compute the DIG
	parser.add_argument('-dataset', 	default=None, type=str) # dataset used, if None SST2 will be used
	parser.add_argument('-strategy', 	default='greedy', choices=['greedy', 'maxcount'], help='The algorithm to find the next anchor point')
	parser.add_argument('-steps', 		default=30, type=int)	# m
	parser.add_argument('-topk', 		default=20, type=int)	# k
	parser.add_argument('-factor', 		default=0, 	type=int)	# f
	parser.add_argument('-knn_nbrs',	default=500, type=int)	# KNN
	parser.add_argument('-seed', 		default=42, type=int)
	parser.add_argument('-output_dir', 	type=str)

	args = parser.parse_args()

	main(args)
