import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *
import matplotlib.pyplot as plt
import sys
from scipy.special import softmax

from transformers import AutoTokenizer, AutoModelForMaskedLM
from normalizer import normalize

tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert_large_generator")
model = AutoModelForMaskedLM.from_pretrained(
    "csebuetnlp/banglabert_large_generator", output_hidden_states=True
)

if torch.cuda.is_available():
    model = model.to("cuda")


def softmax(arr, axis=1):
    e = np.exp(arr)
    return e / e.sum(axis=axis, keepdims=True)


def get_sentence_tokens(sentence):
    input_token = tokenizer(normalize(sentence), return_tensors="pt")
    return input_token


def get_mask_index(input_token, last=False):
    if not last:
        mask_token_index = (input_token.input_ids == tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        if len(mask_token_index > 1):
            mask_token_index = (input_token.input_ids == tokenizer.mask_token_id)[
                0
            ].nonzero(as_tuple=True)[0][:1]
    else:
        mask_token_index = (input_token.input_ids == tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0][-1:]
    return mask_token_index


def get_logits(input_token):
    if torch.cuda.is_available():
        input_token = input_token.to("cuda")
    with torch.no_grad():
        logits = model(**input_token).logits
    return logits


def get_mask_fill_logits(
    sentence: str, words: Iterable[str], use_last_mask=False, apply_softmax=False
) -> Dict[str, float]:
    input_token = get_sentence_tokens(sentence)
    mask_i = get_mask_index(input_token, use_last_mask)
    out_logits = get_logits(input_token).cpu().detach().numpy()
    if apply_softmax:
        out_logits = softmax(out_logits, axis=-1)
    return {w: out_logits[0, mask_i, tokenizer.encode(w)[1]] for w in words}


def get_word_vector(sentence, word):
    normalized_sent = normalize(sentence)
    input_token = tokenizer(normalized_sent, return_tensors="pt")
    sent_list = sentence.split(" ")
    idx = sent_list.index(word) + 1 
    with torch.no_grad():
        outputs = model(**input_token)
        return outputs[1][24][0].detach().cpu().numpy()[idx]


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
