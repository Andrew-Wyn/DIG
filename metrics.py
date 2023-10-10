import numpy as np, pickle
import torch


# Classification Metrics

def eval_log_odds(forward_fn, predict, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=20):
	logits_original						= forward_fn(predict, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	predicted_label						= torch.argmax(logits_original).item()
	prob_original						= torch.softmax(logits_original, dim=0)
	topk_indices						= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	local_input_embed					= input_embed.detach().clone()
	local_input_embed[0][topk_indices]	= mask_token_emb
	logits_perturbed					= forward_fn(predict, local_input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	prob_perturbed						= torch.softmax(logits_perturbed, dim=0)

	return (torch.log(prob_perturbed[predicted_label]) - torch.log(prob_original[predicted_label])).item(), predicted_label


# TODO: CHECK IT !
def eval_anti_log_odds(forward_fn, predict, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=20):
	logits_original									= forward_fn(predict, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	predicted_label									= torch.argmax(logits_original).item()
	prob_original									= torch.softmax(logits_original, dim=0)
	topk_indices									= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	complement_topk_indices							= np.setdiff1d(range(attr.shape[0]), topk_indices)
	local_input_embed								= input_embed.detach().clone()
	local_input_embed[0][complement_topk_indices]	= mask_token_emb
	logits_perturbed								= forward_fn(predict, local_input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	prob_perturbed									= torch.softmax(logits_perturbed, dim=0)

	return (torch.log(prob_perturbed[predicted_label]) - torch.log(prob_original[predicted_label])).item(), predicted_label


def eval_sufficiency(forward_fn, predict, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
	logits_original							= forward_fn(predict, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	predicted_label							= torch.argmax(logits_original).item()
	prob_original							= torch.softmax(logits_original, dim=0)
	topk_indices							= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	if len(topk_indices) == 0:
		# topk% is too less to select even word - so no masking will happen.
		return 0

	mask									= torch.zeros_like(input_embed[0][:,0]).bool()
	mask[topk_indices]						= 1
	masked_input_embed						= input_embed[0][mask].unsqueeze(0)
	masked_attention_mask					= None if attention_mask is None else attention_mask[0][mask].unsqueeze(0)
	masked_position_embed					= None if position_embed is None else position_embed[0][:mask.sum().item()].unsqueeze(0)
	masked_type_embed						= None if type_embed is None else type_embed[0][mask].unsqueeze(0)
	logits_perturbed						= forward_fn(masked_input_embed, attention_mask=masked_attention_mask, position_embed=masked_position_embed, type_embed=masked_type_embed, return_all_logits=True).squeeze()
	prob_perturbed							= torch.softmax(logits_perturbed, dim=0)

	return (prob_original[predicted_label] - prob_perturbed[predicted_label]).item()


def eval_comprehensiveness(forward_fn, predict, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
	logits_original					= forward_fn(predict, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	predicted_label					= torch.argmax(logits_original).item()
	prob_original					= torch.softmax(logits_original, dim=0)
	topk_indices					= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	mask 							= torch.ones_like(input_embed[0][:,0]).bool()
	mask[topk_indices]				= 0
	masked_input_embed				= input_embed[0][mask].unsqueeze(0)
	masked_attention_mask			= None if attention_mask is None else attention_mask[0][mask].unsqueeze(0)
	masked_position_embed			= None if position_embed is None else position_embed[0][:mask.sum().item()].unsqueeze(0)
	masked_type_embed				= None if type_embed is None else type_embed[0][mask].unsqueeze(0)
	logits_perturbed				= forward_fn(masked_input_embed, attention_mask=masked_attention_mask, position_embed=masked_position_embed, type_embed=masked_type_embed, return_all_logits=True).squeeze()
	prob_perturbed					= torch.softmax(logits_perturbed, dim=0)

	return (prob_original[predicted_label] - prob_perturbed[predicted_label]).item()


# Regression Metrics

def regression_eval_log_odds(forward_fn, predict, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=20):
	logits_original						= forward_fn(predict, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	topk_indices						= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	local_input_embed					= input_embed.detach().clone()
	local_input_embed[0][topk_indices]	= mask_token_emb
	logits_perturbed					= forward_fn(predict, local_input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()

	return np.abs((logits_perturbed - logits_original).item())


# TODO: CHECK IT !
def regression_eval_anti_log_odds(forward_fn, predict, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk=20):
	logits_original									= forward_fn(predict, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	topk_indices									= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	complement_topk_indices							= np.setdiff1d(range(attr.shape[0]), topk_indices)
	local_input_embed								= input_embed.detach().clone()
	local_input_embed[0][complement_topk_indices]	= mask_token_emb
	logits_perturbed								= forward_fn(predict, local_input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()

	return np.abs((logits_perturbed - logits_original).item())


def regression_eval_sufficiency(forward_fn, predict, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
	logits_original							= forward_fn(predict, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	topk_indices							= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	if len(topk_indices) == 0:
		# topk% is too less to select even word - so no masking will happen.
		return 0

	mask									= torch.zeros_like(input_embed[0][:,0]).bool()
	mask[topk_indices]						= 1
	masked_input_embed						= input_embed[0][mask].unsqueeze(0)
	masked_attention_mask					= None if attention_mask is None else attention_mask[0][mask].unsqueeze(0)
	masked_position_embed					= None if position_embed is None else position_embed[0][:mask.sum().item()].unsqueeze(0)
	masked_type_embed						= None if type_embed is None else type_embed[0][mask].unsqueeze(0)
	logits_perturbed						= forward_fn(masked_input_embed, attention_mask=masked_attention_mask, position_embed=masked_position_embed, type_embed=masked_type_embed, return_all_logits=True).squeeze()
	
	return np.abs((logits_perturbed - logits_original).item())


def regression_eval_comprehensiveness(forward_fn, predict, input_embed, position_embed, type_embed, attention_mask, base_token_emb, attr, topk=20):
	logits_original					= forward_fn(predict, input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	topk_indices					= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	mask 							= torch.ones_like(input_embed[0][:,0]).bool()
	mask[topk_indices]				= 0
	masked_input_embed				= input_embed[0][mask].unsqueeze(0)
	masked_attention_mask			= None if attention_mask is None else attention_mask[0][mask].unsqueeze(0)
	masked_position_embed			= None if position_embed is None else position_embed[0][:mask.sum().item()].unsqueeze(0)
	masked_type_embed				= None if type_embed is None else type_embed[0][mask].unsqueeze(0)
	logits_perturbed				= forward_fn(predict, masked_input_embed, attention_mask=masked_attention_mask, position_embed=masked_position_embed, type_embed=masked_type_embed, return_all_logits=True).squeeze()
	
	return np.abs((logits_perturbed - logits_original).item())


def eval_wae(scaled_features, word_embedding, epsilon=0.1):
	# compute smallest distance of the embedding to any word embedding and report the average among all words in the path for a sentence
	dists = []
	for emb in scaled_features:
		all_dist = torch.sqrt(torch.sum((word_embedding - emb.unsqueeze(0)) ** 2, dim=1))
		dists.append(torch.min(all_dist).item())

	return np.mean(dists)
