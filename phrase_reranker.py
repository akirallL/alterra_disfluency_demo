from phonemes import get_pronounce_dist
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, RobertaForMaskedLM, RobertaTokenizerFast
from scipy.special import softmax
import torch, math
import numpy as np


def get_score_spec(sentence):
    # print('SENTENCE', sentence)
    with torch.no_grad():
        tokenize_input = tokenizer_spec.tokenize(sentence)
#             print(len(tokenize_input))
        tensor_input = torch.tensor([tokenizer_spec.convert_tokens_to_ids(tokenize_input)]).to(device)
        # print(tensor_input.shape)
        predictions=model_spec(tensor_input)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
#         print(predictions.logits.squeeze().shape,tensor_input.squeeze().shape)
#         print(predictions.logits.shape,tensor_input.shape)
#             print(predictions.logits.shape)
        loss = loss_fct(predictions.logits.squeeze(),tensor_input.squeeze()).cpu().data 
        return math.exp(loss)
#     try:
#         with torch.no_grad():
#             tokenize_input = tokenizer_spec.tokenize(sentence)
# #             print(len(tokenize_input))
#             tensor_input = torch.tensor([tokenizer_spec.convert_tokens_to_ids(tokenize_input)]).to(device)
#             predictions=model_spec(tensor_input)
#             loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
#     #         print(predictions.logits.squeeze().shape,tensor_input.squeeze().shape)
#     #         print(predictions.logits.shape,tensor_input.shape)
# #             print(predictions.logits.shape)
#             loss = loss_fct(predictions.logits.squeeze(),tensor_input.squeeze()).cpu().data 
#             return math.exp(loss)
#     except Exception as ex:
#         print('Exception:', ex)
#         return np.inf


def rerank_by_model(pat, cands):
    res = []
    for c in cands:
        res.append((c, get_score_spec(pat.format(c))))
    res.sort(key=lambda x:x[1])
    return res
    res, _ = zip(*res)
    return list(res)



if 'model_spec' not in locals():
    device = torch.device('cuda')
    model_spec = RobertaForMaskedLM.from_pretrained('roberta-base').eval().to(device)
    tokenizer_spec = RobertaTokenizerFast.from_pretrained('roberta-base')
