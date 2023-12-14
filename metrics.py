import numpy as np, pickle
import torch


# Classification Metrics

def eval_log_odds(ff, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk):
	logits_original						= ff(input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	predicted_label						= torch.argmax(logits_original).item()
	prob_original						= torch.softmax(logits_original, dim=0)
	topk_indices						= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	local_input_embed					= input_embed.detach().clone()
	local_input_embed[0][topk_indices]	= mask_token_emb
	logits_perturbed					= ff(local_input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	prob_perturbed						= torch.softmax(logits_perturbed, dim=0)

	return (torch.log(prob_perturbed[predicted_label]) - torch.log(prob_original[predicted_label])).item(), prob_original[predicted_label].item(), prob_perturbed[predicted_label].item()


# TODO: CHECK IT !
def eval_anti_log_odds(ff, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk):
	logits_original									= ff(input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	predicted_label									= torch.argmax(logits_original).item()
	prob_original									= torch.softmax(logits_original, dim=0)
	topk_indices									= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	complement_topk_indices							= np.setdiff1d(range(attr.shape[0]), topk_indices)
	local_input_embed								= input_embed.detach().clone()
	local_input_embed[0][complement_topk_indices]	= mask_token_emb
	logits_perturbed								= ff(local_input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	prob_perturbed									= torch.softmax(logits_perturbed, dim=0)

	return (torch.log(prob_perturbed[predicted_label]) - torch.log(prob_original[predicted_label])).item(), prob_original[predicted_label].item(), prob_perturbed[predicted_label].item()


# Regression Metrics
def regression_eval_log_odds(ff, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk):
	logits_original						= ff(input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	topk_indices						= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	local_input_embed					= input_embed.detach().clone()
	local_input_embed[0][topk_indices]	= mask_token_emb
	logits_perturbed					= ff(local_input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()

	return np.abs((logits_perturbed - logits_original).item()), logits_original, logits_perturbed


# TODO: CHECK IT !
def regression_eval_anti_log_odds(ff, input_embed, position_embed, type_embed, attention_mask, mask_token_emb, attr, topk):
	logits_original									= ff(input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()
	topk_indices									= torch.topk(attr, int(attr.shape[0] * topk / 100), sorted=False).indices
	complement_topk_indices							= np.setdiff1d(range(attr.shape[0]), topk_indices)
	local_input_embed								= input_embed.detach().clone()
	local_input_embed[0][complement_topk_indices]	= mask_token_emb
	logits_perturbed								= ff(local_input_embed, attention_mask=attention_mask, position_embed=position_embed, type_embed=type_embed, return_all_logits=True).squeeze()

	return np.abs((logits_perturbed - logits_original).item()), logits_original, logits_perturbed