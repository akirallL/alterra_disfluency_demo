import itertools
import numpy as np
from phonemes import get_pronounce_dist, get_top_phoneme_neighbors, pronounce2


def _is_good_neighbor(word_comb: list, neighbor: str, max_good_relative_fraction=0.8):
    word_comb_str = ' '.join(word_comb)
    dst = get_pronounce_dist(word_comb_str, neighbor)
    orig_pron = pronounce2(word_comb_str)[0]
    neigh_pron = pronounce2(neighbor)[0]
    len_frac_orig = dst / len(orig_pron) if orig_pron else 1
    len_frac_neigh = dst / len(neigh_pron) if neigh_pron else 1
    return len_frac_orig <= max_good_relative_fraction or len_frac_neigh <= max_good_relative_fraction


def _max_common_prefix(a, b):
    i = 0
    while(i < len(a) and i < len(b) and a[i] == b[i]):
        i += 1
    return i


def _max_common_substring(a, b):
    max_pref_lens = [_max_common_prefix(a[i:], b) for i in range(len(a))]
    return max(max_pref_lens)


def clever_bubble_sort(candidates, nll_scores, ph_distances, orig_phrase: str, BADLY_RECOGNIZED_WORDS):
    n = len(candidates)
    indices = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            ind_i, ind_j = indices[i], indices[j]

            common_str_len_i = _max_common_substring(candidates[ind_i], orig_phrase)
            common_str_len_j = _max_common_substring(candidates[ind_j], orig_phrase)

            iou_i = common_str_len_i / len(candidates[ind_i])
            iou_j = common_str_len_j / len(candidates[ind_j])

            if candidates[ind_j] in BADLY_RECOGNIZED_WORDS:
                continue
            if ph_distances[ind_i] > ph_distances[ind_j]:
                indices[i], indices[j] = ind_j, ind_i
                continue
            elif iou_i < iou_j:
                indices[i], indices[j] = ind_j, ind_i
                continue
            elif nll_scores[ind_i] > nll_scores[ind_j]:
                indices[i], indices[j] = ind_j, ind_i
                continue
    
    new_candidates, new_nll_scores, new_ph_distances = [], [], []
    for i in range(n):
        ind_i = indices[i]
        new_candidates.append(candidates[ind_i])
        new_nll_scores.append(nll_scores[ind_i])
        new_ph_distances.append(ph_distances[ind_i])
    
    return new_candidates, new_nll_scores, new_ph_distances



def make_pure_client_vocab_neighbors(word_comb: list, CLIENT_WORDS, n_top=5, BADLY_RECOGNIZED_WORDS=()):
    n = len(word_comb)
    combs = []
    # print(word_comb)

    if len(CLIENT_WORDS['special_vocab']) ** n > 50000:
        return []

    for comb_ln in range(max(1, n - 1), n + 1):
        vocabs = [CLIENT_WORDS['special_vocab'] for _ in range(comb_ln)]
        for cmb in itertools.product(*vocabs):
            if max(BADLY_RECOGNIZED_WORDS[w] for w in cmb) >= 0.75:
                continue
            combs.append(cmb)
    combs = list(set(combs))
    # print('get_phoneme_nbrs', len(combs))
    
    nbrs = get_top_phoneme_neighbors(' '.join(word_comb), [' '.join(cmb) for cmb in combs], n_top=20)

    nbrs = [n for n in nbrs if _is_good_neighbor(word_comb, n)]

    return nbrs[:n_top]


def make_conversation_ngram_neihbors(word_comb: list, CLIENT_WORDS, n_top=5, BADLY_RECOGNIZED_WORDS=()):
    n = len(word_comb)
    combs = []
    # print('make ngram nbrs')
    for comb_ln in range(max(0, n - 1), n + 2):
        if comb_ln not in CLIENT_WORDS['conversation_ngrams']:
            continue
        for cmb in CLIENT_WORDS['conversation_ngrams'][comb_ln]:
            if ' '.join(word_comb) in cmb or max(BADLY_RECOGNIZED_WORDS[w] for w in cmb.split()) >= 1:
                continue
            combs.append(cmb)
    combs = list(set(combs))

    if len(combs) > 100000:
        return []

    if not combs:
        return []
    # print('CMBS', len(combs))

    nbrs = get_top_phoneme_neighbors(' '.join(word_comb), combs)
    # print('finish ngram')
    return nbrs[:n_top]


def make_neighbors_flow(word_comb: str, CLIENT_WORDS, n_top=10, BADLY_RECOGNIZED_WORDS=()):
    word_comb = [x.strip() for x in word_comb.split() if x.strip()]
    # print(word_comb)
    client_vocab_nbrs = make_pure_client_vocab_neighbors(word_comb, CLIENT_WORDS, n_top=n_top // 2, BADLY_RECOGNIZED_WORDS=BADLY_RECOGNIZED_WORDS)
    conversation_ngram_nbrs = make_conversation_ngram_neihbors(
        word_comb, 
        CLIENT_WORDS, 
        n_top=max(1, n_top - len(client_vocab_nbrs)),
        BADLY_RECOGNIZED_WORDS=BADLY_RECOGNIZED_WORDS
    )
    return list(set(client_vocab_nbrs + conversation_ngram_nbrs))


def rank_candidates(central_phrase: list, candidates, nll_model_scores, BADLY_RECOGNIZED_WORDS=()):


    phoneme_distances = []
    pronounces = []
    central_phrase_str = ' '.join(central_phrase)
    for w in candidates:
        pronounces.append(pronounce2(w)[0])
        phoneme_distances.append(get_pronounce_dist(central_phrase_str, w))

    n = len(candidates)
    
    cands_and_ph_scores = list(zip(candidates, phoneme_distances))

    cands_and_ph_scores.sort(key=lambda x:x[1])

    return cands_and_ph_scores

    # inds, _ = zip(*cands_and_ph_scores)

    # pluses_for_phonemes = [i / n for i in range(n)]

    # pluses_for_phonemes_positioned = [pluses_for_phonemes[i] for i in inds]
    
    # cands_and_nll_scores = list(zip(range(n), nll_model_scores))

    # cands_and_nll_scores.sort(key=lambda x:x[1], reverse=True)

    # inds, _ = zip(*cands_and_nll_scores)

    # pluses_for_nlls = [i / n for i in range(n)]

    # pluses_for_nlls_positioned = [pluses_for_nlls[i] for i in inds]

    # scores = []
    # indices = list(range(n))
    # for i in range(n):
    #     scores.append(pluses_for_phonemes_positioned[i] + pluses_for_nlls_positioned[i])

    # candidates_and_scores_and_indices = list(zip(candidates, scores, indices))

    # candidates_and_scores_and_indices.sort(key=lambda x:(x[1], x[0], x[2]))

    # candidates, scores, indices = zip(*candidates_and_scores_and_indices)

    # candidates = list(candidates)
    # score_pairs = ['nll={}|ph={}'.format(np.round(pluses_for_nlls_positioned[i], 3), np.round(pluses_for_phonemes_positioned[i], 3)) for i in indices]
    # return list(zip(candidates, score_pairs))


    scores = []

    central_pron = pronounce2(central_phrase_str)

    # for i, (nll_score, pron, ph_dst) in enumerate(zip(nll_model_scores, pronounces, phoneme_distances)):
    #     s = nll_score * (1 - ph_dst / (len(central_pron) + 1e-3))
    #     s = nll_score
    #     scores.append(s)
    
    # cands_and_scores = list(zip(candidates, scores))
    # cands_and_scores.sort(key=lambda x: x[1])
    # # cands_and_scores.sort(reverse=True, key=lambda x: x[1])

    candidates, nll_model_scores, phoneme_distances = clever_bubble_sort(
        candidates, 
        nll_model_scores,
        phoneme_distances,
        orig_phrase=central_phrase_str, 
        BADLY_RECOGNIZED_WORDS=BADLY_RECOGNIZED_WORDS
    )

    return list(zip(candidates, nll_model_scores))

    return cands_and_scores
