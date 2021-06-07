import json
import string
from stop_words import get_stop_words
import wordfreq
import numpy as np
import nltk
from collections import defaultdict, Counter
from nltk.stem import WordNetLemmatizer


def read_tokens(fname):
    if isinstance(fname, list):
        tokens = []
        for fn in fname:
            with open(fn) as f:
                txt = f.read()
                tokens_local = nltk.word_tokenize(txt)
                tokens.extend(tokens_local)
    else:
        with open(fname) as f:
            txt = f.read()
            tokens = nltk.word_tokenize(txt)
    return tokens


def get_n_gram_freq(tokens, n_gram, add_lemmas=True):
    lemmatizer = WordNetLemmatizer()
    counter = Counter()
    ngrams = []
    for i in range(len(tokens) - n_gram + 1):
        slc = tuple(tokens[i:i+n_gram])
        ngrams.append(slc)
        if add_lemmas:
            slc_lemmatized = tuple(lemmatizer.lemmatize(w) for w in slc)
            if slc_lemmatized != slc:
                ngrams.append(slc_lemmatized)
    counter.update(ngrams)
    return counter


def get_context_word_cooccurencies(ngram_2_count, alpha=1):
    word_2_context_word_counts = {}
    for ngram, cnt in ngram_2_count.items():
        center = len(ngram) // 2
        m_w = ngram[center]
        if m_w not in word_2_context_word_counts:
            word_2_context_word_counts[m_w] = {}
        for i in range(0, center):
            c_w = ngram[i]
            if c_w not in word_2_context_word_counts[m_w]:
                word_2_context_word_counts[m_w][c_w] = 0
            score = cnt * (alpha ** abs(center - i))
            word_2_context_word_counts[m_w][c_w] += score
        for i in range(center + 1, len(ngram)):
            c_w = ngram[i]
            if c_w not in word_2_context_word_counts[m_w]:
                word_2_context_word_counts[m_w][c_w] = 0
            score = cnt * (alpha ** abs(center - i))
            word_2_context_word_counts[m_w][c_w] += score
    return word_2_context_word_counts


def get_context_probability_distributions(n_grams, n_th_word):
    frequencies = {
        'previous': {}, 'post': {}, 'full_context': {}
    }
    for ngram, freq in n_grams.items():
        central_word = ngram[n_th_word]
        for k, v in frequencies.items():
            if central_word not in v:
                v[central_word] = {}
        previous = tuple(ngram[:n_th_word])
        post = tuple(ngram[n_th_word+1:])
        full_context = ngram
        frequencies['previous'][central_word][previous] = \
            frequencies['previous'][central_word].get(previous, 0) + freq
        frequencies['post'][central_word][post] = frequencies['post'][central_word].get(post, 0) + freq
        frequencies['full_context'][central_word][full_context] = \
            frequencies['full_context'][central_word].get(full_context, 0) + freq
#     print(frequencies['previous']['bot'])
    for context_type, context_type_dict_freq in frequencies.items():
        for central_word, context_freq in context_type_dict_freq.items():
            N = sum(context_freq.values())
            for context in context_freq:
                context_freq[context] = float(context_freq[context] / N)
    return frequencies


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
    return tfidfs


def get_most_meaningful_words(tokens, n_top, remove_stopwords=False, lemmatize=False):
    meaningful_words = []
    if remove_stopwords:
        tokens = [t for t in tokens if t not in get_stop_words('en')]
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tfidfs = get_tfidf(tokens)
    tfidfs = list(sorted(tfidfs.items(), key=lambda x: x[1], reverse=True))[:n_top]
    return tfidfs


def get_most_special_words(tokens, n_top, remove_stopwords=False, lemmatize=False):
    meaningful_words = []
    if remove_stopwords:
        tokens = [t for t in tokens if t not in get_stop_words('en')]
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tfidfs = get_tokens_freq(tokens)
    tfidfs = list(sorted(tfidfs.items(), key=lambda x: x[1]))[:n_top]
    return tfidfs


def read_meaningful_words(text, n_top=100):
    STOP_WORDS = get_stop_words('en') + ['uh', 'oh', 'eh', 'em', 'hm', 'ummm', 'hmmm', 'ah']
    def jlopen(txt):
        samples = []
        for l in txt.split('\n'):
            l = l.strip()
            if not l:
                continue
            samples.append(json.loads(l))
        return samples
    scribd = jlopen(text)
    recovered_tokens = []
    true_predictions = []
    for s in scribd:
        sent = []
        labels = []
        for w in s['Words']:
            label = int(w['Confidence'] <= -2.)
            tokens = nltk.word_tokenize(w['Word'].lower())

            new_tokens = [t.split('\'') for t in tokens]
            new_tokens = [' \' '.join(tt).split() for tt in new_tokens]
            tokens = sum(new_tokens, [])

            labs = [label for t in tokens]
            sent.extend(tokens)
            labels.extend(labs)
        recovered_tokens.append(sent)
        true_predictions.append(labels)

    joined_tokens = list(filter(lambda x: all(y in string.ascii_letters for y in x) and x not in STOP_WORDS,
                     map(lambda x: x.lower(), sum(recovered_tokens, []))))
    meaningful_words = get_most_special_words(joined_tokens, n_top=n_top)

    meaningful_words, freqs = zip(*meaningful_words)

    return list(meaningful_words)
