import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from io import StringIO
import json, nltk, string
from copy import deepcopy

import streamlit.components.v1 as components

from models import load_highlighter, tokenize_and_align_labels, make_contexted_samples, TensorDataset, glue_tokens, make_highlighted_text
from textwrap import dedent
from lr_fixator import get_most_meaningful_words, read_tokens, get_ngrams_from_client_speech


st.title('disfluency detection app')


components.html('''
<style>
.brightness-90 {
    background-color: rgba(0,255,255,0.9);
}
.brightness-20 {
    background-color: rgba(0,255,255,0.2);
}

with_tooltip {
display: inline-block; /* Строчно-блочный элемент */
position: relative; /* Относительное позиционирование */
}
with_tooltip:hover::after {
content: attr(data-title); /* Выводим текст */
position: absolute; /* Абсолютное позиционирование */
left: 20%; top: 30%; /* Положение подсказки */
z-index: 1; /* Отображаем подсказку поверх других элементов */
background: rgba(255,255,230,0.9); /* Полупрозрачный цвет фона */
font-family: Arial, sans-serif; /* Гарнитура шрифта */
font-size: 11px; /* Размер текста подсказки */
padding: 5px 10px; /* Поля */
border: 1px solid #333; /* Параметры рамки */
}
  
</style>
<i class="brightness-90 with_tooltip" data-title="Fuck you 20!">Some Text 90</i>
<i class="brightness-20 with_tooltip" data-title="Fuck you 90!">Some Text 20</i>
''')


def read_jsonl(text):
    def jlopen(txt):
        samples = []
        for l in txt.split('\n'):
            l = l.strip()
            if not l:
                continue
            samples.append(json.loads(l))
        return samples
    scribd = jlopen(text)
    tokens = []
    confidences = []
    for s in scribd:
        toks_locls = []
        conf_local = []
        for w in s['Words']:
            toks_locls.append(w['Word'].lower().strip(string.punctuation))
            conf_local.append(w['Confidence'])
        tokens.append(toks_locls)
        confidences.append(conf_local)
    return tokens, confidences


model_pack = load_highlighter('./models/distilbert_with_confidences_transformer.pth')

THRESCHOLD = float(st.number_input(label='Sensivity', value=0.5, min_value=0.0, max_value=1.0))

include_conversation_vocabulary = st.checkbox('Include conversation vocabulary')

client_vocab_default = dedent('''
alterra phraser manychat chatfuel zendesk colibri query covid
intent bot nlp rnn api semantically answer corpus messenger
coronavirus name webhook automate faq timestamp slack integer 
template demo paloalto
''')


txt = st.text_area(
    label='special vocab', 
    value=client_vocab_default
)

CLIENT_WORDS = {'special_vocab': [w.strip() for w in txt.split() if w.strip()]}
CLIENT_WORDS['total_vocabulary'] = deepcopy(CLIENT_WORDS['special_vocab'])

alterrra_tokens = read_tokens('./models/texts.txt')
HELP_VOCAB =  get_most_meaningful_words(alterrra_tokens, n_top=500)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    # bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    string_data = stringio.read()

    tokens, confidences = read_jsonl(string_data)

    if include_conversation_vocabulary:
        # CLIENT_WORDS['conversation_unigrams'] = get_ngrams_from_client_speech(sum(tokens, []), 1)
        # CLIENT_WORDS['conversation_bigrams'] = get_ngrams_from_client_speech(sum(tokens, []), 2)
        CLIENT_WORDS['conversation_ngrams'] = {}
        for i in range(1, 6):
            CLIENT_WORDS['conversation_ngrams'][i] = get_ngrams_from_client_speech(sum(tokens, []), i)
        # print(CLIENT_WORDS['conversation_ngrams'][3])
        for k in CLIENT_WORDS['conversation_ngrams'].keys():
            CLIENT_WORDS['total_vocabulary'].extend(CLIENT_WORDS['conversation_ngrams'][k])
        CLIENT_WORDS['total_vocabulary'] = list(set(CLIENT_WORDS['total_vocabulary']))

    tokenized_inputs = tokenize_and_align_labels({'tokens': tokens, 'confidences': confidences}, model_pack.tok)

    from torch.utils.data.dataloader import DataLoader
    import torch
    device = torch.device('cuda')

    inputs_with_context = make_contexted_samples(tokenized_inputs, 1, max_input_length=256)

    test_dataset = TensorDataset(inputs_with_context, tok=model_pack.tok, max_length=256)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    test_logits = []
    total_eval_loss = 0.0
    for batch in test_dataloader:
            batch = {k:v.to(device) for k, v in batch.items()}
            with torch.no_grad():        
                outputs = model_pack.model(**batch)
                loss, logits = outputs.loss, outputs.logits
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

    #         # Move logits and labels to CPU
            logits = torch.softmax(logits, dim=2).detach().cpu().numpy()
            test_logits.append(logits)

    test_logits = np.vstack(test_logits)
    test_logits = test_logits[:, :, 1] #test_logits.argmax(axis=2)

    response = {
        'input_ids': [],
        'tokens': [],
        'predictions': []
    }

    for i, s in enumerate(inputs_with_context):
        toks, preds = s.match_result_with_predictions(test_logits[i].flatten())
        response['input_ids'].append(toks)
        response['tokens'].append([model_pack.tok.decode([t]) for t in toks])
        response['predictions'].append(preds)

    result = {
        'tokens': [],
        'labels': []
    }

    # for i, s in enumerate(inputs_with_context[:5]):
    #     print([model_pack.tok.decode([t]) for t in s.main_context['input_ids']])
    # print('-'*20)
    # for i, s in enumerate(inputs_with_context[:7]):
    #     print([model_pack.tok.decode([t]) for t in s.tokenization_result['input_ids']])

    for i in range(len(response['tokens'])):
        glued_toks, glued_labs = glue_tokens(response['tokens'][i], response['predictions'][i])
        result['tokens'].append(glued_toks)
        result['labels'].append(glued_labs)

    if include_conversation_vocabulary:
        st.text_area(
            label='joined conversation and client vocab', 
            value='\t'.join(list(sorted(CLIENT_WORDS['total_vocabulary'])))
        )

    html = make_highlighted_text(result['tokens'], result['labels'], CLIENT_WORDS=CLIENT_WORDS, HELP_VOCAB=HELP_VOCAB, THRESCHOLD=THRESCHOLD)
    components.html(html, scrolling=True, height=600, width=1200)
    # print(html)
