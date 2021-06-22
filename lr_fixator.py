import subprocess as sp
import os
from glob import glob
from os import listdir as lsd
from os.path import join as pj
from os.path import basename as bn
from tqdm import tqdm
import json
import string
from time import sleep
import wordfreq
from stop_words import get_stop_words
from bisect import bisect_left
import numpy as np
from collections import Counter
from copy import deepcopy
from sklearn.model_selection import train_test_split
import pandas as pd
from phonemes import pronounce2, full_vocabulary, LevenshteinDist, confusion, get_top_phoneme_neighbors, STOP_WORDS
import nltk, random
from Levenshtein import distance as word_distance
from sklearn.linear_model import LogisticRegression
import itertools


LOAD_MODEL = False


def good_word(w):
    good_chars = string.ascii_letters + '\''
    return all(x in good_chars for x in w)


def clean_word(w):
    w = w.strip(string.digits + string.punctuation)
    return w

def read_tokens(fname):
    toks = []
    with open(fname) as fin:
        for l in fin:
            sent = [clean_word(w) for w in l.split() if w]
            sent = [w.lower() for w in sent if w and good_word(w)]
            toks.extend(sent)
    return toks


def get_ranges_from_array(arr, predicate, range_predicate):
    l = 0
    N = len(arr)
    result = []
    while l < N:
        if not predicate(arr[l]):
            l += 1
        else:
            r = l + 1
            while r < N and predicate(arr[r]):
                r += 1
            if range_predicate(arr[l:r]):
                result.append((l, r))
            l = r
    return result


def get_most_meaningful_words(tokens, remove_stopwords=False, n_top=200):
    def word_tfidf(w, term_freq):
        log_idf = -np.log(wordfreq.word_frequency(w, 'en') + 1e-10)
        tfidf = term_freq * log_idf
        return tfidf
    def get_tfidf(tokens):
        counter = Counter(tokens)
        N = sum(counter.values())
        freqs = {w: counter[w] / N for w in counter}
        tfidfs = {w:word_tfidf(w, freqs[w]) for w in freqs}
        return tfidfs
    def get_tokens_freq(tokens):
        counter = Counter(tokens)
        N = sum(counter.values())
        freqs = {w: counter[w] / N for w in counter}
        tfidfs = {w:wordfreq.word_frequency(w, 'en') for w in freqs}
        return tfidfs, freqs
    
    meaningful_words = []
    if remove_stopwords:
        tokens = [t for t in tokens if t not in get_stop_words('en')]
    tfidfs_map = get_tfidf(tokens)
    tfidfs = list(sorted(tfidfs_map.items(), key=lambda x: x[1], reverse=True))[:n_top]

    freqs, client_freqs = get_tokens_freq(tokens)
    freqs = list(sorted(freqs.items(), key=lambda x: x[1]))[:n_top]
    
    tfidf_w, tfidf_s = list(zip(*tfidfs))
    freq_w, freq_s = list(zip(*freqs))
    all_w = list(set(freq_w) | set(tfidf_w))
    
    result = {}
    for w in all_w:
        item = {
            'client_tfidf': tfidfs_map[w],
            'client_freq': client_freqs[w]
        }
        result[w] = item
    
    return result


def get_score(sentence):
    try:
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
        predictions=bertMaskedLM(tensor_input)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
#         print(predictions.logits.squeeze().shape,tensor_input.squeeze().shape)
#         print(predictions.logits.shape,tensor_input.shape)
        loss = loss_fct(predictions.logits.squeeze(),tensor_input.squeeze()).cpu().data 
        return math.exp(loss)
    except:
        return None

def gather_predictions_for_sentences(sentences, batch_size=128, tokenizer_batch_size=512, max_length=128):
    N = len(sentences)
    tensor_input = []
    for i in tqdm(list(range(0, N, tokenizer_batch_size))):
        tensor_input.append(
        tokenizer(sentences[i:i+tokenizer_batch_size], 
                  return_tensors='pt', 
                  padding='max_length',
                  max_length=max_length)['input_ids'].to(device))
    tensor_input = torch.vstack(tensor_input)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
#     tokenize_input = tokenizer(sentences, return_tensors='pt', padding='max_length', max_length=128)
#     tokenize_input['input_ids'] = tokenize_input['input_ids'].to(device)
    all_predictions = []
    with torch.no_grad():
        for i in tqdm(list(range(0, N, batch_size))):
#             tensor_batch = tokenizer(sentences[i:i+batch_size], 
#                                      return_tensors='pt', padding='max_length', max_length=128)
            tensor_batch = tensor_input[i:i+batch_size]
            predictions = bertMaskedLM(tensor_batch.to(device))
            n = tensor_batch.shape[0]
            for j in range(n):
                pad_ind = (tensor_batch[j] == tokenizer.pad_token_id).int().argmax()
                if pad_ind == 0:
                    pad_ind = max_length
                tokenized_sent = tensor_batch[j, 1:pad_ind - 1]
                logits = predictions.logits[j, 1:pad_ind - 1]
#                 print(tokenized_sent.shape, logits.shape, tokenized_sent)
                raw_pred = loss_fct(logits, tokenized_sent).cpu().data
                all_predictions.append(math.exp(raw_pred))
    return all_predictions
    
def pronounce_sentence(sent: list):
    res = []
    for w in sent:
        res.extend(pronounce2(w, full_vocabulary['pronounciations'])[0])
    return res

def get_pronounce_dist_between_sentences(sent1, sent2):
    pron1 = pronounce_sentence(sent1)
    pron2 = pronounce_sentence(sent2)
    return LevenshteinDist(pron1, pron2, 1, 1, confusion, 1)[-1][-1]


if LOAD_MODEL: # or 'BertForMaskedLM' not in locals():
    from transformers import BertForMaskedLM, BertTokenizer, BertTokenizerFast
    import torch, math
    device = torch.device('cuda')
#     bertMaskedLM = BertForMaskedLM.from_pretrained(
#         '/home/akiralll/PycharmProjects/bert_mlm/distilbert-base-uncased-train_wiki_articles_lm-train_youtube/')
    bertMaskedLM = BertForMaskedLM.from_pretrained('distilbert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('distilbert-base-uncased')
    bertMaskedLM.to(device)
    


def make_features_for_candidate(
    orig_text: list, 
    candidate_text: list, 
    l_context_index, 
    r_context_index, 
    client_vocab=None):
    orig_text_str = ''.join(orig_text)
    candidate_text_str = ''.join(candidate_text)
    orig_text_pronounce = pronounce_sentence(orig_text)
    cand_text_pronounce = pronounce_sentence(candidate_text)
    
    features = [
        {
            'name': 'CHAR_DISTANCE',
            'value': word_distance(orig_text_str, candidate_text_str)
        },
        {
            'name': 'CHAR_DISTANCE_/_LEN_ORIG_TEXT',
            'value': word_distance(orig_text_str, candidate_text_str) / len(orig_text_str)
        },
        {
            'name': 'CHAR_DISTANCE_/_LEN_CAND_TEXT',
            'value': word_distance(orig_text_str, candidate_text_str) / len(candidate_text_str)
        },
        {
            'name': 'PHONEME_DISTANCE',
            'value': get_pronounce_dist_between_sentences(orig_text, candidate_text_str)
        },
        {
            'name': 'PHONEME_DISTANCE_/_LEN_ORIG_PRONOUNCE',
            'value': get_pronounce_dist_between_sentences(orig_text, candidate_text) \
                    / (len(orig_text_pronounce) + 1e-3)
        },
        {
            'name': 'PHONEME_DISTANCE_/_LEN_CAND_PRONOUNCE',
            'value': get_pronounce_dist_between_sentences(orig_text, candidate_text) \
                    / (len(cand_text_pronounce) + 1e-3)
        },
        {
            'name': 'LEN_ORIG_PRONOUNCE',
            'value': len(orig_text_pronounce)
        },
        {
            'name': 'LEN_CAND_PRONOUNCE',
            'value': len(cand_text_pronounce)
        },
        {
            'name': 'MEAN_FREQUENCY_OF_ORIG_WORD_BY_WORDFREQ',
            'value': np.mean([wordfreq.word_frequency(w, 'en') for w in orig_text]),
        },
        {
            'name': 'MEAN_FREQUENCY_OF_CAND_WORD_BY_WORDFREQ',
            'value': np.mean([wordfreq.word_frequency(w, 'en') for w in candidate_text]),
        },
        {
            'name': 'MEAN_ORIG_PHRASE_TFIDF_IN_CLIENT_VOCAB',
            'value': np.mean([client_vocab.get(w, {'client_tfidf': 0})['client_tfidf'] for w in orig_text]),
        },
        {
            'name': 'MEAN_CAND_PHRASE_TFIDF_IN_CLIENT_VOCAB',
            'value': np.mean([client_vocab.get(w, {'client_tfidf': 0})['client_tfidf'] for w in candidate_text]),
        },
        {
            'name': 'MEAN_ORIG_PHRASE_FREQ_IN_CLIENT_VOCAB',
            'value': np.mean([client_vocab.get(w, {'client_freq': 0})['client_freq'] for w in orig_text]),
        },
        {
            'name': 'MEAN_CAND_PHRASE_FREQ_IN_CLIENT_VOCAB',
            'value': np.mean([client_vocab.get(w, {'client_freq': 0})['client_freq'] for w in candidate_text]),
        },
        {
            'name': 'CANDIDATE_CONTAINS_DIGITS',
            'value': int(any(x in string.digits for x in candidate_text_str)),
        },
        {
            'name': 'CANDIDATE_CONTAINS_PUNCTUATION',
            'value': int(any(x in string.punctuation for x in candidate_text_str)),
        },
        {
            'name': 'CANDIDATE_IS_STOPWORD',
            'value': int(candidate_text_str in STOP_WORDS)
        },
        {
            'name': 'ORIG_PHRASE_IS_STOPWORD',
            'value': int(orig_text_str in STOP_WORDS)
        }
    ]
    return features


def make_context_features_cb_samples(cb_samples):
    # gather statistics
    group_index_2_stat = {}
    for s in cb_samples:
        gi = s['group_index']
        if gi not in group_index_2_stat:
            group_index_2_stat[gi] = {
                'candidate_substituted_scores': [],
                'original_sentence_score': None
            }
        group_index_2_stat[gi]['original_sentence_score'] = s['orig_sentence_score']
        group_index_2_stat[gi]['candidate_substituted_scores'].append(s['candidate_substituted_sentence_score'])
    
    # agg_stats
    for i, stat in group_index_2_stat.items():
        stat['candidate_substituted_scores_mean'] = np.mean(stat['candidate_substituted_scores'])
        stat['candidate_substituted_scores_std'] = np.std(stat['candidate_substituted_scores']) + 1e-6
        stat['candidate_substituted_scores_min'] = np.min(stat['candidate_substituted_scores'])
        stat['candidate_substituted_scores_max'] = np.max(stat['candidate_substituted_scores'])
    
    for s in cb_samples:
        gi = s['group_index']
        s['features'].extend([
            {
                'name': 'CANDIDATE_SUBSTITUTED_SENTENCE_SCORE',
                'value': s['candidate_substituted_sentence_score']
            },
            {
                'name': 'CANDIDATE_SUBSTITUTED_SENTENCE_SCORE_-_MEAN_/_STD',
                'value': (s['candidate_substituted_sentence_score'] - \
                        group_index_2_stat[gi]['candidate_substituted_scores_mean']) / \
                        group_index_2_stat[gi]['candidate_substituted_scores_std']
            },
            {
                'name': 'CANDIDATE_SUBSTITUTED_SENTENCE_SCORE_-_MIN_/_(MAX-MIN)',
                'value': (s['candidate_substituted_sentence_score'] - \
                        group_index_2_stat[gi]['candidate_substituted_scores_min']) / \
                        (group_index_2_stat[gi]['candidate_substituted_scores_max'] - \
                         group_index_2_stat[gi]['candidate_substituted_scores_min'] + 1e-6)
            },
            {
                'name': '(CANDIDATE_SUBSTITUTED_SENTENCE_SCORE_-_MEAN_/_STD)_-_(ORIG_SENTENCE_SCORE_-_MEAN_/_STD)',
                'value': (s['candidate_substituted_sentence_score'] - s['orig_sentence_score']) / \
                        group_index_2_stat[gi]['candidate_substituted_scores_std']
            },
            {
                'name': 'CANDIDATE_SUBSTITUTED_SENTENCE_SCORE_-_ORIG_SENTENCE_SCORE',
                'value': (s['candidate_substituted_sentence_score'] - s['orig_sentence_score'])
            },
            {
                'name': 'CANDIDATE_SUBSTITUTED_SENTENCE_SCORE_/_LEN_CANDIDATE_CONTEXT',
                'value': s['candidate_substituted_sentence_score'] / len(s['context_with_substituted_candidate'])
            },
            {
                'name': 'ORIG_SENTENCE_SCORE_/_LEN_ORIG_CONTEXT',
                'value': s['orig_sentence_score'] / len(s['context'])
            },
            {
                'name': 'LEN_ORIG_CONTEXT',
                'value': len(s['context'])
            },
            {
                'name': 'LEN_CANDIDATE_SUBSTITUTED_CONTEXT',
                'value': len(s['context_with_substituted_candidate'])
            },
            {
                'name': 'LEN_CANDIDATE_SUBSTITUTED_CONTEXT_/_LEN_ORIG_CONTEXT',
                'value': len(s['context_with_substituted_candidate']) / (len(s['context']) + 1e-6)
            }
        ])
    return cb_samples


def prepare_samples_for_cb(samples, hard_coded_candidates=None):
    cb_samples = []
    
    for index, s in enumerate(tqdm(samples)):
        source = deepcopy(s['tokens'])
        target = deepcopy(s['gt_tokens'])
        asr_ranges = get_ranges_from_array(s['verbose_labels'], lambda x: x != 0, lambda x: any(y != 2 for y in x))
        gt_ranges = get_ranges_from_array(s['gt_tokens_labels'], lambda x: x != 0, lambda x: True)
        assert len(asr_ranges) == len(gt_ranges)
        for rg_s, rg_t in zip(asr_ranges, gt_ranges):
            l_s, r_s = rg_s
            l_t, r_t = rg_t
            badly_recognized_text = ' '.join(source[l_s:r_s])
            target_text = ' '.join(target[l_t:r_t])
#             print(badly_recognized_text, '->', target_text)
            if hard_coded_candidates is None:
                if random.random() < 0.5:
                    podskazki = get_top_phoneme_neighbors(
                        badly_recognized_text, set(list(s['help_vocab'].keys())) - {target_text}, n_top=10)
                else:
                    podskazki = set(random.sample(list(s['help_vocab'].keys()), 
                                                  min(10, len(s['help_vocab'])))) - {target_text}
                    podskazki = list(podskazki)
            else:
                podskazki = hard_coded_candidates
            for p in podskazki:
                sample_features = make_features_for_candidate(
                    orig_text=source[l_s:r_s],
                    candidate_text=p.split(),
                    l_context_index=l_s,
                    r_context_index=r_s,
                    client_vocab=s['help_vocab']
                )
                cb_samples.append({
                    'features': sample_features,
                    'target': 0,
                    'badly_recognized_text': badly_recognized_text,
                    'candidate_text': p,
                    'context': source,
                    'context_with_substituted_candidate': source[:l_s] + p.split() + source[r_s:],
                    'group_index': index,
                    # 'target_text': target_text
                })
            true_target_text_features = make_features_for_candidate(
                orig_text=source[l_s:r_s],
                candidate_text=target[l_t:r_t],
                l_context_index=l_s,
                r_context_index=r_s,
                client_vocab=s['help_vocab']
            )
            cb_samples.append({
                'features': true_target_text_features,
                'target': 1,
                'badly_recognized_text': badly_recognized_text,
                'candidate_text': target_text,
                'context': source,
                'context_with_substituted_candidate': source[:l_s] + target_text.split() + source[r_s:],
                'group_index': index,
                # 'target_text': target_text
            })
    
    # group_index_2_orig_sentence = {}
    # for sample in cb_samples:
    #     group_index_2_orig_sentence[sample['group_index']] = sample['context']
    # group_indexes, orig_sentences = list(zip(*list(sorted(group_index_2_orig_sentence.items()))))
    # orig_sentence_predictions = gather_predictions_for_sentences([' '.join(s) for s in orig_sentences])
    # group_index_2_orig_sentence_score = dict(zip(group_indexes, orig_sentence_predictions))
    
    # candidate_substituted_sentences = []
    # for sample in cb_samples:
    #     candidate_substituted_sentences.append(' '.join(sample['context_with_substituted_candidate']))
    
    # candidate_substituted_sentences_predictions = gather_predictions_for_sentences(
    #                 candidate_substituted_sentences)
    
    # for i, sample in enumerate(cb_samples):
    #     sample['orig_sentence_score'] = group_index_2_orig_sentence_score[sample['group_index']]
    #     sample['candidate_substituted_sentence_score'] = \
    #             candidate_substituted_sentences_predictions[i]
    
    # samples = make_context_features_cb_samples(cb_samples)
    
    return cb_samples


def make_test_sample(sent: list, wrong_word_index_l, wrong_word_index_r, help_vocab, candidates: list):
    samples = []
    l_t = wrong_word_index_l
    r_t = wrong_word_index_r
    gt_tokens = sent
    gt_tokens_labels = [0 if i < l_t or i >= r_t else 1 for i in range(len(gt_tokens))]
    for c in candidates:
        cand_tokens = c.split()
        l_s = l_t
        r_s = l_s + len(cand_tokens)
        tokens = sent[:l_s] + cand_tokens + sent[r_s:]
        vebose_labels = [0 if i < l_s or i >= r_s else 1 for i in range(len(tokens))]
        sample = {
            'tokens': tokens,
            'gt_tokens': gt_tokens,
            'verbose_labels': vebose_labels,
            'gt_tokens_labels': gt_tokens_labels,
            'help_vocab': help_vocab
        }
        samples.append(sample)
    
    features = prepare_samples_for_cb(samples, candidates)
    data = {
        k['name']: [] for k in features[0]['features']
    }
    data['target'] = []
    data['badly_recognized_text'] = []
    data['candidate_text'] = []
    # data['target_text'] = []
    data['context'] = []
    random.shuffle(features)
    for f in features:
        for k in f['features']:
            data[k['name']].append(k['value'])
        for k in ['target', 'badly_recognized_text', 'candidate_text', 'context',]:
            data[k].append(f[k])
    df = pd.DataFrame(data=data)
    return df


def make_test_sample_2(word_comb, candidates, help_vocab):
    cb_samples = []
    for p in candidates:
        badly_recognized_tokens = word_comb.split()
        sample_features = make_features_for_candidate(
                        orig_text=badly_recognized_tokens,
                        candidate_text=p.split(),
                        l_context_index=0,
                        r_context_index=len(badly_recognized_tokens),
                        client_vocab=help_vocab
                    )
        cb_samples.append({
            'features': sample_features,
            'target': 0,
            'badly_recognized_text': word_comb,
            'candidate_text': p,
            'context': badly_recognized_tokens,
            'context_with_substituted_candidate': p.split(),
        })
    cb_samples.append({
            'features': sample_features,
            'target': 1,
            'badly_recognized_text': word_comb,
            'candidate_text': word_comb,
            'context': badly_recognized_tokens,
            'context_with_substituted_candidate': badly_recognized_tokens,
        })
    features = cb_samples
    data = {
        k['name']: [] for k in features[0]['features']
    }
    data['target'] = []
    data['badly_recognized_text'] = []
    data['candidate_text'] = []
    data['context'] = []
    random.shuffle(features)
    for f in features:
        for k in f['features']:
            data[k['name']].append(k['value'])
        for k in ['target', 'badly_recognized_text', 'candidate_text', 'context']:
            data[k].append(f[k])
    df = pd.DataFrame(data=data)
    return df


def evaluate_sample(sample, central_text):
    train_columns = ['CANDIDATE_CONTAINS_DIGITS', 'CANDIDATE_CONTAINS_PUNCTUATION', 'CANDIDATE_IS_STOPWORD', 
    'CHAR_DISTANCE', 'CHAR_DISTANCE_/_LEN_CAND_TEXT', 'CHAR_DISTANCE_/_LEN_ORIG_TEXT', 'LEN_CAND_PRONOUNCE', 
    'LEN_ORIG_PRONOUNCE', 'MEAN_CAND_PHRASE_FREQ_IN_CLIENT_VOCAB', 'MEAN_CAND_PHRASE_TFIDF_IN_CLIENT_VOCAB',
     'MEAN_FREQUENCY_OF_CAND_WORD_BY_WORDFREQ', 'MEAN_FREQUENCY_OF_ORIG_WORD_BY_WORDFREQ', 
     'MEAN_ORIG_PHRASE_FREQ_IN_CLIENT_VOCAB', 'MEAN_ORIG_PHRASE_TFIDF_IN_CLIENT_VOCAB', 
     'ORIG_PHRASE_IS_STOPWORD', 'PHONEME_DISTANCE', 'PHONEME_DISTANCE_/_LEN_CAND_PRONOUNCE', 
     'PHONEME_DISTANCE_/_LEN_ORIG_PRONOUNCE']
    non_feature_columns = ['target', 'badly_recognized_text', 'candidate_text', 'context',]
    WEIGHTS = np.array([[ 0.00000000e+00,  1.19336941e+00, -9.37956263e-01,
        -9.10860368e-01, -4.40045761e-01, -4.56662605e-02,
         2.95188685e-01,  1.39309686e-01, -2.00982855e-02,
         8.42150367e-01, -1.83510718e-02,  1.41584803e-02,
        -2.58391711e-02, -1.73674143e-01,  3.30475285e-01,
         1.14180462e-01,  4.28573585e-04,  3.97560332e-04]])
    BIAS = np.array([-1.24675658])
    X_test = sample[train_columns].to_numpy()
    clf = LogisticRegression()
    clf.coef_ = WEIGHTS
    clf.intercept_ = BIAS
    ts_preds = clf.predict_proba(X_test)[:, 1]
    df_to_display = sample[non_feature_columns].copy()
    df_to_display['score'] = ts_preds
    df_to_display = df_to_display.head(50).sort_values(by='score', ascending=False)
    items = []
    for i in range(df_to_display.shape[0]):
        items.append((df_to_display.candidate_text.iloc[i], df_to_display.score.iloc[i]))
    items = list(set(items))
    items.sort(key=lambda x: x[1], reverse=True)
    return items


# def make_n_gram_candidates(word_combination: str, candidates: list, n, n_top=20):
#     ngram_vocs = [candidates for _ in range(n)]
#     ngram_candidates = []
#     for ngram_word_comb in itertools.product(*ngram_vocs):
#         if len(set(ngram_word_comb)) < len(ngram_word_comb):
#             continue
#         ngram_candidates.append(' '.join(ngram_word_comb))
#     result = get_top_phoneme_neighbors(word_combination, ngram_candidates, n_top=n_top)
#     return result


def get_ngrams_from_client_speech(tokens: list, ngram):
    ngrams = set()
    for i in range(0, len(tokens) - ngram  + 1):
        ngrams.add(' '.join(tokens[i:i+ngram]))
    return list(ngrams)

