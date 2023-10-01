import os
import json
import sys, numpy as np, argparse, random
sys.path.append('../')
from random import shuffle
from collections import defaultdict

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
from roberta_helper import nn_init, load_mappings, nn_forward_func, get_mask_token_emb, get_inputs, get_tokens

all_outputs = []

# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, get_tokens):
	# computes the attributions for given input

	# move inputs to main device
	inp = [x.to(device) if x is not None else None for x in inputs]

	# compute attribution
	scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp
	attr = run_dig_explanation(attr_func, scaled_features, position_embed, type_embed, attention_mask, (2**args.factor)*(args.steps+1)+1)

	# compute metrics
	log_odd, _		= eval_log_odds(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)
	anti_log_odd, _ = eval_anti_log_odds(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)
	comp			= eval_comprehensiveness(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)
	suff			= eval_sufficiency(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)

	return log_odd, anti_log_odd, comp, suff


def regression_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, get_tokens):
	# computes the attributions for given input

	# move inputs to main device
	inp = [x.to(device) if x is not None else None for x in inputs]

	# compute attribution
	scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp
	attr = run_dig_explanation(attr_func, scaled_features, position_embed, type_embed, attention_mask, (2**args.factor)*(args.steps+1)+1)

	# compute metrics
	log_odd			= regression_eval_log_odds(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)
	anti_log_odd 	= regression_eval_anti_log_odds(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)
	comp			= regression_eval_comprehensiveness(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)
	suff			= regression_eval_sufficiency(nn_forward_func, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=args.topk)

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


def sentiment_ita_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, get_tokens, xai_metrics):

	"""		
	log_odd, anti_log_odd, comp, suff = classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, get_tokens)
	
	xai_metrics["log_odd"] += log_odd
	xai_metrics["anti_log_odd"] += anti_log_odd
	xai_metrics["comp"] += comp
	xai_metrics["suff"] += suff
	"""

	# TODO: IMPLEMENT COLLECTING THE SCORES FOR EACH SUB_TASK (POS | NEG)

	pass


def sst2_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, get_tokens, xai_metrics):

		log_odd, anti_log_odd, comp, suff = classification_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, get_tokens)
		
		xai_metrics["log_odd"] += log_odd
		xai_metrics["anti_log_odd"] += anti_log_odd
		xai_metrics["comp"] += comp
		xai_metrics["suff"] += suff


def complexity_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, get_tokens, xai_metrics):

		log_odd, anti_log_odd, comp, suff = regression_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, get_tokens)
		
		xai_metrics["reg_log_odd"] += log_odd
		xai_metrics["reg_anti_log_odd"] += anti_log_odd
		xai_metrics["reg_comp"] += comp
		xai_metrics["reg_suff"] += suff


def main(args):

	# set seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	auxiliary_data = load_mappings(args.task, args.modeltype, args.runtype, knn_nbrs=args.knn_nbrs)

	# Fix the gpu to use
	device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# init model and tokenizer in cpu first
	nn_init(device, args.modelname)

	# Define the Attribution function
	attr_func = DiscretetizedIntegratedGradients(nn_forward_func)

	if args.dataset is None:
		# load the dataset
		dataset	= load_dataset('sst2')['test']
		data	= list(zip(dataset['sentence'], dataset['label'], dataset['idx']))
	else:
		if args.task == "complexity":
			dataset = read_complexity_dataset(args.dataset)
			data = list(zip(dataset["text"], dataset["label"]))
		else: # TODO: add italian sentiment analysis loading
			pass

	shuffle(data)

	# get ref token embedding
	mask_token_emb = get_mask_token_emb(device)

	# compute the DIG attributions for all the inputs
	print('Starting attribution computation...')
	inputs = []

	xai_metrics = defaultdict(lambda: 0)

	count=0

	print_step = 10

	for row in tqdm(data[:600]):
		inp = get_inputs(row[0], device)
		input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask = inp
		scaled_features 		= monotonic_paths.scale_inputs(input_ids.squeeze().tolist(), ref_input_ids.squeeze().tolist(),\
											device, auxiliary_data, steps=args.steps, factor=args.factor, strategy=args.strategy)
		inputs					= [scaled_features, input_ids, ref_input_ids, input_embed, ref_input_embed, position_embed, ref_position_embed, type_embed, ref_type_embed, attention_mask]

		if args.task == "complexity": # call regression metrics
			complexity_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, get_tokens, xai_metrics)
		elif args.task == "sentiment_en": # call classification metrics
			sst2_calculate_attributions(inputs, device, args, attr_func, mask_token_emb, nn_forward_func, get_tokens, xai_metrics)
		elif args.task == "sentiment_it": # call classification metrics twice one for each sub-task
			pass

		count += 1

		# print the metrics
		if count % print_step == 0:
			print(xai_metrics)

	print(xai_metrics)

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	with open(f"{args.output_dir}/xai_metrics.json", 'w') as f:
		json.dump(xai_metrics, f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='IG Path')
	parser.add_argument('-modelname', 	default="xlm-roberta-base", type=str)
	parser.add_argument('-modeltype', 	default="roberta", type=str)
	parser.add_argument('-task', 		default="complexity", type=str)
	parser.add_argument('-runtype', 	default="hf_np", type=str)
	parser.add_argument('-dataset', 	default=None, type=str)
	parser.add_argument('-strategy', 	default='greedy', choices=['greedy', 'maxcount'], help='The algorithm to find the next anchor point')
	parser.add_argument('-steps', 		default=30, type=int)	# m
	parser.add_argument('-topk', 		default=20, type=int)	# k
	parser.add_argument('-factor', 		default=0, 	type=int)	# f
	parser.add_argument('-knn_nbrs',	default=500, type=int)	# KNN
	parser.add_argument('-seed', 		default=42, type=int)
	parser.add_argument('-output_dir', 	type=str)

	args = parser.parse_args()

	main(args)
