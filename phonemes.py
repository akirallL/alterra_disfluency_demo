import re
from string import digits
import numpy as np
from os.path import join as pj
import json


with open('config.json') as fl:
    config = json.load(fl)

cmupath = config['cmu_dict_path']
cmudict_file= pj(cmupath, "cmudict-0.7b.json")
cmusymbols_file = pj(cmupath, "cmudict-0.7b.symbols")

def add_vocab_line(line, vocab):
    if line.startswith(";;;"):
        return
    entry,pronounciation = line.strip().split("  ",2)
    index=len(vocab['entries'])
    vocab['entries'].append(entry)

    m=re.match("^(.+)(\(\d\))",entry)
    if m:
        word=m.group(1)
    else:
        word=entry
    vocab['words'].append(word)

    pronounciation=pronounciation.split(" ")
    if word in vocab['pronounciations']:
        vocab['pronounciations'][word].append(pronounciation)
        vocab['indices'][word].append(index)
    else:
        vocab['pronounciations'][word]=[pronounciation]
        vocab['indices'][word]=[index]
    
    pos=vocab['root']
    for c in pronounciation:
        if c not in pos:
            pos[c]={}
            pos[c]['']=pos[c]
        pos=pos[c]
    if ' ' in pos:
        pos[' '].append(index)
    else:
        pos[' ']=[index]

def init_vocab():
    vocab={'words': [],
           'entries': [],
           'indices': {},
           'pronounciations': {},
           'root': {}
          }
    vocab['root']['']=vocab['root']
    vocab['root'][' ']=vocab['root']
    return(vocab)

def append_vocab(dictionary, vocab, present_words):
    for line in dictionary:
        w, *a = line.split()
        if w not in present_words:
            continue
        add_vocab_line(line, vocab)

def build_vocab(present_words, *dicts):
    v=init_vocab()
    for d in dicts:
        append_vocab(d, v, present_words)
    return(v)


import wordfreq
from stop_words import get_stop_words
STOP_WORDS = get_stop_words('en')

wf = list(sorted(wordfreq.get_frequency_dict('en').items(), key=lambda x: -x[1]))
present_words, _ = zip(*wf[:15000])
common_present_words = set(x.upper() for x in present_words)

with open(cmudict_file) as cmu_dict_file_desc:
    full_vocabulary = build_vocab(common_present_words, cmu_dict_file_desc)


def pronounce(string, pronounciations):
    ps=[[]]
    for w in string.upper().split():
        pn=[]
        for p in ps:
            if p:
                p.append(' ')
            for pi in pronounciations[w]:
                pn.append(p+pi)
        ps=pn
    return(pn)

import nltk, random
ALPHABET = nltk.corpus.cmudict.dict()


def wordbreak(s):
    if len(s) >= 40:
        return None
    s = s.lower()
    if s in ALPHABET:
        return ALPHABET[s]
    middle = len(s) / 2
    partition = sorted(list(range(len(s))), key=lambda x: (x - middle) ** 2 - x)
    for i in partition:
        pre, suf = (s[:i], s[i:])
        if pre in ALPHABET and wordbreak(suf) is not None:
            return [random.choice(ALPHABET[pre]) + random.choice(wordbreak(suf))]
            return [x + y for x, y in product(ALPHABET[pre], wordbreak(suf))]
    return None


def pronounce2(string, pronounciations=None):
    words = string.split()
    prons = [wordbreak(w) for w in words]
    if any(p is None for p in prons):
        big_word = ''.join(words)
        pron = wordbreak(big_word)
        if pron is None:
            return [[]]
        return pron
    big_pron = []
    for p in prons:
        big_pron.extend(p[0])
    return [big_pron]


confusion_groups=[
    [0.5, 'TH', 'DH', 'Z','S'],
    [0.5,'AA','OW','AO'],
    [1,'AA','AE','AH','AO','AW','AY','EH','ER','EY','IH','IY','OW','OY','UH','UW','R'],
    [0.5,'AE','AY','EH','EY'],
    [0.5, 'B', 'P'],
    [0.5, 'D', 'T'],
    [0.5, 'G', 'HH', 'JH'],
    [0.5, 'V', 'F']
]


confusion_groups = [
    [0.5, 'TH', 'DH', 'Z', 'S', 'D', 'T'],
    [0.5, 'AA', 'OW', 'AO'],
    [1, 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'R'],
    [0.5, 'AE', 'AY', 'EH', 'EY'],
    [0.5, 'B', 'P'],
    [0.5, 'G', 'HH', 'JH', 'K'],
#     [0.5, 'G', 'K'],
#     [0.5, 'V', 'F',],
#     [0.5, 'V', 'W',],
    [0.5, 'N', 'NG'],
    [0.5, 'CH', 'SH'],
#     [0.5, 'C', 'T'],
]

def mkconfusion():
    global confusion
    global alphabet

    confusion={'':{' ':.5}, ' ':{' ':0, '':.5}, 'EOS':{' ':0}}
    alphabet=[]
    digits="0123456789"

    for line in open(cmusymbols_file):
        if line:
            alphabet.append(line.strip())  
    
    for c1 in alphabet:
        if c1 not in confusion:
            confusion[c1]={}
        for c2 in alphabet:
            if c1==c2:
                confusion[c1][c2]=0
            else:
                if c1.rstrip(digits) == c2.rstrip(digits):
                    confusion[c1][c2]=0.1
                else:
                    confusion[c1][c2]=2
        confusion[c1][' ']=2
        confusion[c1]['']=2
        confusion[' '][c1]=2
        confusion[''][c1]=2

    cg_amended=[]
    for g in confusion_groups:
        g_amended=g.copy()
        for h in g[1:]:
            for c in alphabet:
                if c not in g_amended and h.rstrip(digits) == c.rstrip(digits):
                    g_amended.append(c)
        cg_amended.append(g_amended)

    for g in cg_amended:
        for h1 in g[1:]:
            for h2 in g[1:]:
                if confusion[h1][h2] > g[0]:
                    confusion[h1][h2]=g[0]
mkconfusion()


MAX_BEAMS=500

# beams = [beam1, beam2, ...]
# beam = {'pos':pos, 'digested':digested}
# pos = root['h']['e']['l']...
# digested = [ interpretation1, interpretation2]
# interpretation = [tokens, loss] ***assumed to be already sorted***
# tokens = [int, int, int, ...]

def digest1(context, symbol, trie, maxloss):
    beams=[]
    if symbol not in confusion:
        confusion[symbol]={symbol:0}
    for c,l in confusion[symbol].items():
        for b in context:
            if c in b['pos']:
                b=b.copy()
                b['digested']=b['digested'].copy()
                b_c=b['pos'][c]
                if type(b_c) is list:
                    b['digested']=[[tokens+[b_c1],loss+l] 
                                   for b_c1 in b_c for tokens,loss in b['digested'] if (loss+l)<=maxloss]
                    b['pos']=trie
                    if len(b['digested']):
                        beams.append(b)
                else:
                    if b['digested'][0][1]+l<=maxloss:
                        b['digested']=[[tokens,loss+l] for tokens,loss in b['digested']]
                        b['pos']=b_c
                        beams.append(b)
    return(sorted(beams,key=lambda x: x['digested'][0][1])[:MAX_BEAMS])

def digest(beams, symbol, trie, maxloss):
    beams_new=[]
    beams_hungry=beams
    
    while beams_hungry:
        beams_new+=digest1(beams_hungry, symbol, trie, maxloss)
        beams_new.sort(key=lambda x: x['digested'][0][1])
        del(beams_new[MAX_BEAMS:])
        beams_hungry=digest1(beams_hungry, '', trie, maxloss)

        if len(beams_hungry)+len(beams_new)>MAX_BEAMS:
            jbeams=beams_hungry+beams_new
            argsort_jbeams=sorted(range(len(jbeams)), key=lambda x: jbeams[x]['digested'][0][1])
            tail=0
            for i in range(MAX_BEAMS-1,-1,-1):
                j=argsort_jbeams[i]
                if j<len(beams_hungry) and beams_hungry[j]['digested'][0][1]<=maxloss:
                    tail=j+1
                    break
            del(beams_hungry[tail:])

    beam_by_pos={}
    beams_merged=[]
    for b in beams_new:
        i=id(b['pos'])
        if i in beam_by_pos:
            beam_by_pos[i]['digested'].extend(b['digested'])          
            beam_by_pos[i]['digested'].sort(key=lambda x: x[1])
            del beam_by_pos[i]['digested'][MAX_BEAMS:]
        else:
            beam_by_pos[i]=b
            beams_merged.append(b)

    for b in beams_merged:
        dupes=set()
        digested=[]
        for i in b['digested']:
            s=str(i[0])
            if s not in dupes:
                dupes.add(s)
                digested.append(i)
        b['digested']=digested

    return(beams_merged)

def flatten(beams):
    flat_beams={}
    
    for b in beams:
        for i in b['digested']:
            s=tuple(i[0])
            if s not in flat_beams or flat_beams[s][1]>i[1]:
                flat_beams[s]=i

    return(sorted(flat_beams.values(), key=lambda x: x[1]))

def process(string, trie, maxloss):
    beams=[{'pos': trie, 'digested':[[[],0]]}]
    
    for c in string:
        beams=digest(beams, c, trie, maxloss)
    beams=digest(beams, 'EOS', trie, maxloss)

    beams=flatten(beams)
            
    return(beams)

def spell_process(string, vocabulary, maxloss):
#     sound=pronounce(string, full_vocabulary['pronounciations'])
    sound=pronounce2(string, vocabulary['pronounciations'])
    
    beams=[]
    
    for s in sound:
        beam=process(s, vocabulary['root'], maxloss)
        for b in beam:
            bs=" ".join([vocabulary['words'][w] for w in b[0]])
            beams.append([bs,b[1]])
    
    flat_beams={}
    
    for b in beams:
        if b[0] not in flat_beams or flat_beams[b[0]][1]>b[1]:
            flat_beams[b[0]]=b

    return(sorted(flat_beams.values(), key=lambda x: x[1]))


def spell_process_with_dynamic_threshold(string, vocabulary, max_error_fraction=0.7):
#     sound=pronounce(string, full_vocabulary['pronounciations'])
    sound=pronounce2(string, vocabulary['pronounciations'])
    
    mean_sound_len = np.mean([len(s) for s in sound])
    maxloss = max(1, int(max_error_fraction * mean_sound_len))
    print(mean_sound_len, maxloss)
    
    beams=[]
    
    for s in sound:
        beam=process(s, vocabulary['root'], maxloss)
        for b in beam:
            bs=" ".join([vocabulary['words'][w] for w in b[0]])
            beams.append([bs,b[1]])
    
    flat_beams={}
    
    for b in beams:
        if b[0] not in flat_beams or flat_beams[b[0]][1]>b[1]:
            flat_beams[b[0]]=b

    return(sorted(flat_beams.values(), key=lambda x: x[1]))


def LevenshteinDist(S, T, delete_cost, insert_cost, replace_cost_mtx, transpose_cost):
    S = [x.rstrip(digits) for x in S if x.rstrip(digits)]
    T = [x.rstrip(digits) for x in T if x.rstrip(digits)]
    M = len(S)
    N = len(T)
    d = [[0 for j_ in range(N + 1)] for i_ in range(M + 1)]
    d[0][0] = 0
    for i in range(1, M + 1):
        d[i][0] = d[i - 1][0] + delete_cost
    for j in range(1, N + 1):
        d[0][j] = d[0][j - 1] + insert_cost
    
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if S[i - 1] == T[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = d[i - 1][j - 1] + replace_cost_mtx[S[i - 1]][T[j - 1]]
            
            d[i][j] = min(d[i][j], d[i - 1][j] + delete_cost, d[i][j - 1] + insert_cost)
            
            if i > 1 and j > 1 and S[i - 1] == T[j - 2] and S[i - 2] == T[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + transpose_cost)
    return d


def get_pronounce_dist(w1, w2):
    pron1 = pronounce2(w1, full_vocabulary['pronounciations'])[0]
    pron2 = pronounce2(w2, full_vocabulary['pronounciations'])[0]
    return LevenshteinDist(pron1, pron2, 1, 1, confusion, 1)[-1][-1]


def get_closest_words_by_phoneme_levenshtein(spelled_word, 
                                             word_list, 
                                             delete_cost=1, 
                                             insert_cost=1,
                                             replace_cost_mtx=confusion, 
                                             transpose_cost=1):
#     pron = pronounce(spelled_word, full_vocabulary['pronounciations'])[0]
    pron = pronounce2(spelled_word, full_vocabulary['pronounciations'])[0]
    items = []
    for w in word_list:
        if not any(x in full_vocabulary['pronounciations'] for x in [w.lower(), w.upper()]):
            continue
        w_pron = pronounce(w.lower(), full_vocabulary['pronounciations'])[0]
        dist = LevenshteinDist(pron, w_pron, 
                               insert_cost=insert_cost, 
                               delete_cost=delete_cost,
                               replace_cost_mtx=replace_cost_mtx, 
                               transpose_cost=transpose_cost)[-1][-1]
        item = {
            'word': w,
            'pron': w_pron,
            'dist': dist
        }
        items.append(item)
    items.sort(key=lambda x: x['dist'])
    return items


def get_top_phoneme_neighbors(sent, words, n_top=10):
    sw = [(get_pronounce_dist(w.lower(), sent.lower()), w.lower()) for w in words]
    sw.sort()
    _, res = zip(*sw[:n_top])
    return list(res)
