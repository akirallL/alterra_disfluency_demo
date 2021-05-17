import os
from string import punctuation
import warnings
from transformers import DataCollatorForTokenClassification
import torch
import json
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, BertTokenizer, DistilBertTokenizer
import numpy as np
from scipy.special import softmax
import nltk
import streamlit as st


# class bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKCYAN = '\033[96m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'


warnings.simplefilter('ignore')

bad_punct = []  # [x for x in punctuation if x != '\'']


def tokenize_text(text):
    text = text.strip().lower()
    tokens = nltk.word_tokenize(text)
    return tokens


def replace_multi(txt, symbs):
    for s in symbs:
        txt = txt.replace(s, '')
    return txt


def prepare_tokens(txt):
    txt = replace_multi(txt, bad_punct)
    txt = txt.strip().lower()
    tokens = tokenizer(txt)
    return tokens


def slice_tokens(tokens, max_window_size=100, step=10):
    slices = []
    tokens = tokens['input_ids'][1:-1]  # remove bos/eos
    positions = list(range(len(tokens)))
    sliced_positions = []
    start_pos = 0

    at_the_end = False

    while start_pos < len(tokens) - max_window_size + 1 or not at_the_end:
        if start_pos >= len(tokens) - max_window_size + 1:
            at_the_end = True
        lb = start_pos
        ub = min(start_pos + max_window_size, len(tokens))
        while lb >= 0:
            if not tokenizer.decode([tokens[lb]])[0].startswith('#'):
                break
            lb -= 1
        while ub < len(tokens):
            if not tokenizer.decode([tokens[ub - 1]])[0].endswith('#'):
                break
            ub += 1
        tokens_slice = [101] + tokens[lb:ub] + [102]
        attention_mask = [1 for i in range(len(tokens_slice))]
        slices.append({
            'input_ids': tokens_slice,
            'attention_mask': attention_mask,
            'labels': [0 for i in range(len(tokens_slice))]
        })
        sliced_positions.append(positions[lb:ub])
        start_pos += step
    return slices, sliced_positions


def read_text(fname, delim='.', remove_punct=True):
    with open(fname) as f:
        text = f.read().lower()
    sentences = text.split(delim)
    if remove_punct:
        for i in range(len(sentences)):
            sentences[i] = sentences[i].strip().strip(punctuation).strip()
    sentences = [s for s in sentences if s]

    return sentences


def process_text(text, delim='.', remove_punct=True):
    text = text.lower()
    sentences = text.split(delim)
    if remove_punct:
        for i in range(len(sentences)):
            sentences[i] = sentences[i].strip().strip(punctuation).strip()
    sentences = [s for s in sentences if s]

    return sentences


from string import digits, ascii_lowercase
from copy import deepcopy


def glue_tokens(tokens, suspictious_indicators):
    new_tokens, new_labels = [], []

    for token, label_idx in zip(tokens, suspictious_indicators):
        if not token.startswith("_"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
            new_labels[-1] = new_labels[-1] or label_idx
        else:
            new_labels.append(label_idx)
            new_tokens.append(token)
    return new_labels, new_tokens


def expand_ranges(tokens, indicators):
    N = len(tokens)
    previous_indicators = deepcopy(indicators)
    for i, (tok, ind) in enumerate(zip(tokens, previous_indicators)):
        if ind == 1 and tok in punctuation and tok != '\'':
            indicators[i] = 0
        elif tok == '\'':
            print(tokens[i - 1:i + 2])
            repl = False
            if i > 0 and all(x in ascii_lowercase for x in tokens[i - 1]):
                indicators[i - 1] = 1
                repl = True
            if i + 1 < N and tokens[i + 1] in ['ll', 's', 've', 't', 'm', 're']:
                indicators[i + 1] = 1
                repl = True
            if repl:
                indicators[i] = 1

    for i, (tok, ind) in enumerate(zip(tokens, previous_indicators)):
        if i > 0 and i + 1 < N and previous_indicators[i - 1] == 1 and previous_indicators[i + 1] == 1 and \
                tok not in punctuation and indicators[i] == 0 and tok == '\'':
            indicators[i] = 1

    return tokens, indicators


def make_masked_text(tokens, indicators):
    l, r = 0, 0
    N = len(tokens)
    new_tokens = []
    original_phrases = []
    while l < N:
        if indicators[l] == 0:
            new_tokens.append(tokens[l])
            l += 1
        else:
            r = l
            while r < N and indicators[r] == 1:
                r += 1
            new_tokens.append('<mask>')
            original_phrases.append(tokens[l:r])
            l = r
    text = ' '.join(new_tokens)
    return text, original_phrases


class SentenceWithContext:
    def __init__(self, sentence, left_context, right_context):
        self.sentence = sentence
        self.left_context = left_context
        self.right_context = right_context

    def apply_tokenization(self, tokenizer):
        tokenization_result = {'input_ids': [], 'attention_mask': []}
        sentence_tokens = tokenizer(self.sentence)
        if self.left_context:
            left_context_tokens = tokenizer(self.left_context)
        else:
            left_context_tokens = dict.fromkeys(tokenization_result.keys(),
                                                [tokenizer.cls_token_id, tokenizer.sep_token_id])
        if self.right_context:
            right_context_tokens = tokenizer(self.right_context)
        else:
            right_context_tokens = dict.fromkeys(tokenization_result.keys(),
                                                 [tokenizer.cls_token_id, tokenizer.sep_token_id])

        for tokenizer_outputs in [left_context_tokens, sentence_tokens, right_context_tokens]:
            for k, v in tokenization_result.items():
                v.extend(tokenizer_outputs[k][1:-1])

        self.left_border = len(left_context_tokens['input_ids']) - 1
        self.right_border = self.left_border + len(sentence_tokens['input_ids']) - 2

        tokenization_result['input_ids'] = \
            [tokenizer.cls_token_id] + tokenization_result['input_ids'] + [tokenizer.sep_token_id]
        tokenization_result['labels'] = [0 for _ in range(len(tokenization_result['input_ids']))]
        tokenization_result['attention_mask'] = [1] + tokenization_result['attention_mask'] + [1]
        self.tokenization_result = tokenization_result
        return self.tokenization_result

    def match_result_with_predictions(self, predictions):
        self.true_tokens = self.tokenization_result['input_ids'][self.left_border:self.right_border]
        self.true_predictions = predictions[self.left_border:self.right_border]
        return self.true_tokens, self.true_predictions


def apply_set_of_trainers(trainers_collection, tokens_batch, sentences_with_contexts):
    central_tokens, true_predictions = None, None
    for trainer, threshold in trainers_collection:
        raw_predictions, labels, _ = trainer.predict(tokens_batch)
        # predictions = np.argmax(raw_predictions, axis=2)
        raw_preds_sf = softmax(raw_predictions, axis=2)
        predictions = (raw_preds_sf[:, :, 1] > threshold).astype(np.int)

        # Remove ignored index (special tokens)
        true_predictions_ = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        central_tokens_ = [s.match_result_with_predictions(true_predictions_[i])
                          for i, s in enumerate(sentences_with_contexts)]
        central_tokens_, true_predictions_ = zip(*central_tokens_)
        if central_tokens is None and true_predictions is None:
            central_tokens = central_tokens_
            true_predictions = true_predictions_
        else:
            for i in range(len(true_predictions_)):
                for j in range(len(true_predictions_[i])):
                    if true_predictions_[i][j] == 'wrong_word':
                        true_predictions[i][j] = 'wrong_word'
    return central_tokens, true_predictions


@st.cache
def download_model(url, filename):
    import requests
    response = requests.get(url)

    totalbits = 0
    if response.status_code == 200:
        with open(os.path.join('models', filename), 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    totalbits += 1024
                    print("Downloaded", totalbits * 1025, "KB...")
                    f.write(chunk)


model_checkpoint = 'distilbert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

data_collator = DataCollatorForTokenClassification(tokenizer)

label_list = ['none', 'wrong_word']

# versions = [
#     ('models/distilbert_spoiled_ner_09_05__00_38.pth', 0.5),
#     ('models/distilbert_base_cased_10pcnt_missspel_medium_dataset_tuned_state_dict_222.pth', 0.5),
# #     'trained_models/distilbert_spoiled_ner_08_05__13_23.pth',
# ]

with open('config.json') as fl:
    config = json.load(fl)
    versions = config['versions']

trainers = []
for version_info in versions:
    download_model(version_info['link'], version_info['name'])
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    args = TrainingArguments(
        "test-wwd",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model,
        args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    model.load_state_dict(
        torch.load(os.path.join('models', version_info['name']), map_location=config['device']))
    model = model.eval()
    trainers.append((trainer, version_info['threshold']))
