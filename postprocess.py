import string
from string import digits, ascii_lowercase, punctuation
from copy import deepcopy
from stop_words import get_stop_words
import nltk
from itertools import product
import random
from itertools import zip_longest

alphabet = nltk.corpus.cmudict.dict()


STOPWORDS = get_stop_words('en') + ['ah', 'oh', 'eh', 'um', 'uh', 'one\'s', 'it\'ll', 'whatever', 'he\'ll']


def wordbreak(s):
#     if len(s) >= 20:
#         return None
    print('S:', s)
    s = s.lower()
    if s in alphabet:
        return alphabet[s]
    middle = len(s) / 2
    partition = sorted(list(range(len(s))), key=lambda x: (x - middle) ** 2 - x)
    for i in partition:
        pre, suf = (s[:i], s[i:])
        if pre in alphabet and wordbreak(suf) is not None:
            return [random.choice(alphabet[pre]) + random.choice(wordbreak(suf))]
            return [x + y for x, y in product(alphabet[pre], wordbreak(suf))]
    return None


def make_sample(sent, labels):
    def normal_word_to_replace(w):
        if any(x not in string.ascii_letters + ' ' for x in w):
            return False
        if w in STOPWORDS:
            return False
        return True
    indexes = [i for i in range(len(sent)) if labels[i] != 0]
    ranges = []
    idx = 0
    k = len(indexes)
    while idx < k:
        l = idx
        r = idx + 1
        while r < k and indexes[r] == indexes[l] + (r - l):
            r += 1
        idx = r
        ranges.append((indexes[l], indexes[r - 1] + 1))
    
    sample = {
        'target': ' '.join(sent),
        'source': deepcopy(sent)
    }
    
    
    
    for rg in reversed(ranges):
        l, r = rg
        spelling = []
        if any(not normal_word_to_replace(w) for w in sample['source'][l:r]):
            continue
        for w in sample['source'][l:r]:
            print(w)
            spell = wordbreak(w)[0]
            spelling.extend(spell)
        sample['source'][l:r] = ['#'] + spelling + ['#']
    
    sample['source'] = ' '.join(sample['source'])
    return sample


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _clip_stopwords(spoiled_word_combination, stop_words):
    tokens = spoiled_word_combination.split()
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == '\'':
            print(i, tokens[i-1:i+2])
            tokens = tokens[:i - 1] + [tokens[i - 1] + tokens[i] + tokens[i + 1]] + tokens[i + 2:]

    if all(t == tokens[0] for t in tokens) and len(tokens) > 1:
        return True, '', tokens[0], ''
    elif all(t in stop_words for t in tokens):
        return True, '', ' '.join(tokens), ''

    for i1 in range(len(tokens)):
        if tokens[i1] not in stop_words:
            break

    prefix = ' '.join(tokens[:i1])
    for i2 in range(len(tokens) - 1, -1, -1):
        if tokens[i2] not in stop_words:
            break
    suffix = ' '.join(tokens[i2 + 1:])

    return False, prefix, ' '.join(tokens[i1:i2 + 1]), suffix


def string_norm(s, symbols=string.punctuation):
    s = s.lower()
    replace_punctuation = str.maketrans(symbols, ' ' * len(symbols))
    s = s.translate(replace_punctuation)
    return s.strip()


def glue_tokens(tokens, suspictious_indicators):
    new_tokens, new_labels = [], []

    for token, label_idx in zip(tokens, suspictious_indicators):
        if token.startswith("##"):
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


def make_highlighted_tokens(tokens, indicators):
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
            while r < N and indicators[r] == 1 or tokens[max(0, r - 1)] == '\'':
                r += 1
            is_all_stopwords, pref, med, suf = _clip_stopwords(' '.join(tokens[l:r]), STOPWORDS)
            if not is_all_stopwords:
                new_tokens.append('{} <strong>{}</strong> {}'.format(pref, med, suf))
            else:
                new_tokens.append('{} {} {}'.format(pref, med, suf))
            original_phrases.append(tokens[l:r])
            l = r
    text = ' '.join(new_tokens)
    return text, original_phrases


def make_highlighted_tokens_2(tokens, indicators):
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
            is_all_stopwords, pref, med, suf = _clip_stopwords(' '.join(tokens[l:r]), STOPWORDS)
            if not is_all_stopwords:
                new_tokens.append(bcolors.BOLD + '{} {} {}'.format(pref, med, suf) + bcolors.ENDC)
            else:
                new_tokens.append('{} {} {}'.format(pref, med, suf))
            original_phrases.append(tokens[l:r])
            l = r
    text = ' '.join(new_tokens)
    return text, original_phrases


def remove_stopword_highlighted_tokens(tokens, indicators):
    l, r = 0, 0
    N = len(tokens)
    new_tokens = []
    new_labels = []
    while l < N:
        if indicators[l] == 0:
            new_tokens.append(tokens[l])
            new_labels.append(0)
            l += 1
        else:
            r = l
            while r < N and (indicators[r] == 1 or tokens[r] == '\'' or tokens[r-1] == '\''):
                r += 1
            is_all_stopwords, pref, med, suf = _clip_stopwords(' '.join(tokens[l:r]), STOPWORDS)
            if not is_all_stopwords:
                pref_tokens = pref.split()
                for tok in pref_tokens:
                    new_tokens.append(tok)
                    new_labels.append(0)
                
                med_tokens = med.split()
                for tok in med_tokens:
                    new_tokens.append(tok)
                    new_labels.append(1)

                suff_tokens = suf.split()
                for tok in suff_tokens:
                    new_tokens.append(tok)
                    new_labels.append(0)
            else:
                for idx in range(l, r):
                    new_tokens.append(tokens[idx])
                    new_labels.append(0)
            l = r
    return new_tokens, new_labels


def make_postprocessed_tokens(tokens, labels):
    all_tokens = sum([x for x in tokens], [])
    indicators = sum([x for x in labels], [])
    indicators = [int(x == 'wrong_word') for x in indicators]
    indicators, all_tokens = glue_tokens(all_tokens, indicators)
    all_tokens, indicators = expand_ranges(all_tokens, indicators)
    masked_text, original_phrases = make_highlighted_tokens(all_tokens, indicators)
    return masked_text, original_phrases


def make_postprocessed_tokens_2(tokens, labels):
    indicators = [[int(x == 'wrong_word') for x in sent_labels] for sent_labels in labels]
    all_original_phrases = []
    all_masked_texts = []
    test_samples_4_genererative_model = []
    for idx in range(len(tokens)):
        indicators_local, tokens_local = glue_tokens(tokens[idx], indicators[idx])
        tokens_local, indicators_local = expand_ranges(tokens_local, indicators_local)
        test_samples_4_genererative_model.append(make_sample(*remove_stopword_highlighted_tokens(tokens_local, indicators_local)))
        masked_text_local, original_phrases_local = make_highlighted_tokens(tokens_local, indicators_local)
        all_original_phrases.extend(original_phrases_local)
        all_masked_texts.append(masked_text_local)
    
    import json
    with open('runtime_results/input.json', 'w') as f:
        json.dump(test_samples_4_genererative_model, f)

    return '<br>'.join(all_masked_texts), all_original_phrases


def make_postprocessed_tokens_3(tokens, labels):
    indicators = [[int(x == 'wrong_word') for x in sent_labels] for sent_labels in labels]
    all_original_phrases = []
    all_masked_texts = []
    for idx in range(len(tokens)):
        indicators_local, tokens_local = glue_tokens(tokens[idx], indicators[idx])
        tokens_local, indicators_local = expand_ranges(tokens_local, indicators_local)
        masked_text_local, original_phrases_local = make_highlighted_tokens_2(tokens_local, indicators_local)
        all_original_phrases.extend(original_phrases_local)
        all_masked_texts.append(masked_text_local)

    return '\n'.join(all_masked_texts), 


def make_postprocessed_tokens_with_asr_signal(tokens, labels, tokens_asr, labels_asr):
    indicators = [[int(x == 'wrong_word') for x in sent_labels] for sent_labels in labels]
    all_original_phrases = []
    all_masked_texts = []
    test_samples_4_genererative_model = []
    for idx in range(len(tokens)):
        indicators_local, tokens_local = glue_tokens(tokens[idx], indicators[idx])
        # print('LEN', len(tokens_local), len(tokens_asr[idx]))
        # print('TT', tokens_local)
        # indicators_local = labels_asr[idx]
        tokens_local, indicators_local = expand_ranges(tokens_local, indicators_local)
        test_samples_4_genererative_model.append(make_sample(*remove_stopword_highlighted_tokens(tokens_local, indicators_local)))
        masked_text_local, original_phrases_local = make_highlighted_tokens(tokens_local, indicators_local)
        all_original_phrases.extend(original_phrases_local)
        all_masked_texts.append(masked_text_local)
    
    import json
    with open('runtime_results/input.json', 'w') as f:
        json.dump(test_samples_4_genererative_model, f)

    return '<br>'.join(all_masked_texts), all_original_phrases


def save_masked_tokens(tokens, labels):
    indicators = [[int(x == 'wrong_word') for x in sent_labels] for sent_labels in labels]
    all_original_phrases = []
    all_masked_texts = []
    for idx in range(len(tokens)):
        indicators_local, tokens_local = glue_tokens(tokens[idx], indicators[idx])
        tokens_local, indicators_local = expand_ranges(tokens_local, indicators_local)
        masked_text_local, original_phrases_local = make_masked_text(tokens_local, indicators_local)
        all_original_phrases.extend(original_phrases_local)
        all_masked_texts.append(masked_text_local)

    import json
    with open('/home/akiralll/PycharmProjects/alterra_cands_consumers/data/dummy_replacement_input_3.txt', 'w') as f:
        f.write('\n'.join(all_masked_texts))
    with open('/home/akiralll/PycharmProjects/alterra_cands_consumers/data/mask_2_spoiled_word_dummy_3.json', 'w') as f:
        json.dump(all_original_phrases, f)


def prettify_html(html):
    pref = '<p style="color: green; opacity: 0.80;">'
    suf = '</p>'
    return pref + html + suf
