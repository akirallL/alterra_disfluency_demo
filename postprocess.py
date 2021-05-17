import string
from string import digits, ascii_lowercase, punctuation
from copy import deepcopy
from stop_words import get_stop_words

STOPWORDS = get_stop_words('en') + ['ah', 'oh', 'eh', 'um', 'uh']


def _clip_stopwords(spoiled_word_combination, stop_words):
    tokens = spoiled_word_combination.split()
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == '\'':
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
            # print(tokens[i - 1:i + 2])
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
            while r < N and indicators[r] == 1:
                r += 1
            is_all_stopwords, pref, med, suf = _clip_stopwords(' '.join(tokens[l:r]), STOPWORDS)
            if not is_all_stopwords:
                new_tokens.append('<strong>{} {} {}</strong>'.format(pref, med, suf))
            else:
                new_tokens.append('{} {} {}'.format(pref, med, suf))
            original_phrases.append(tokens[l:r])
            l = r
    text = ' '.join(new_tokens)
    return text, original_phrases


def make_postprocessed_tokens(tokens, labels):
    all_tokens = sum([x for x in tokens], [])
    indicators = sum([x for x in labels], [])
    indicators = [int(x == 'wrong_word') for x in indicators]
    indicators, all_tokens = glue_tokens(all_tokens, indicators)
    all_tokens, indicators = expand_ranges(all_tokens, indicators)
    masked_text, original_phrases = make_highlighted_tokens(all_tokens, indicators)
    return masked_text, original_phrases
