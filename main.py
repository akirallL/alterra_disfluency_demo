import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from io import StringIO

import streamlit.components.v1 as components

from disfluency_detector import *
from postprocess import *
from fixator_gen import *


st.title('disfluency detection app')

# st.write("Here's our first attempt at using data to create a table:")
# st.write(pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# }))
#
# chart_data = pd.DataFrame(
#      np.random.randn(20, 3),
#      columns=['a', 'b', 'c'])
#
# st.line_chart(chart_data)
#
#
# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])
#
# st.map(map_data)
#
#
# if st.checkbox('Show dataframe'):
#     chart_data = pd.DataFrame(
#        np.random.randn(20, 3),
#        columns=['a', 'b', 'c'])
#
#     chart_data


# option = st.selectbox(
#    'Which number do you like best?',
#     df['first column'])
#
# 'You selected: ', option


def predict_text(text, format='txt'):
    if format == 'txt':
        sentences = process_text(text, '\n', remove_punct=False)
    elif format == 'jsonl':
        sentences, recovered_tokens_asr, true_predictions_asr = read_jsonl(text)

    sentences_from_sides_gap = 5
    N = len(sentences)

    sentences_with_contexts = []
    for i in range(len(sentences)):
        left_border = max(0, i - sentences_from_sides_gap)
        right_border = min(N, i + sentences_from_sides_gap + 1)
        left_context = ' '.join(sentences[left_border:i])
        right_context = ' '.join(sentences[i + 1:right_border])
        print('Left_context:', left_context)
        print('Central part:', sentences[i])
        print('RightContext:', right_context)
        print('\n\n')
        sentences_with_contexts.append(SentenceWithContext(sentences[i], left_context, right_context))

    # tokens_batch = [tokenizer(s) for s in sentences]
    tokens_batch = [s.apply_tokenization(tokenizer) for s in sentences_with_contexts]

    central_tokens, true_predictions = apply_set_of_trainers(trainers, tokens_batch, sentences_with_contexts)

    recovered_tokens = [[tokenizer.decode([x]) for x in t] for t in central_tokens]



    output_lines = []
    for idx in range(len(recovered_tokens)):
        assert len(recovered_tokens[idx]) == len(true_predictions[idx])
        bpe_tokens = []
        for i in range(len(recovered_tokens[idx])):
            if true_predictions[idx][i] == 'wrong_word':
                bpe_tokens.append('<strong>' + recovered_tokens[idx][i] + '</strong>')
            else:
                bpe_tokens.append(recovered_tokens[idx][i])
        print(' '.join(bpe_tokens))
        output_lines.append(' '.join(bpe_tokens))
        print('\n\n')
    
    if format == 'txt':
        return '<p>' + '\n'.join(output_lines) + '</p>', recovered_tokens, true_predictions
    elif format == 'jsonl':
        return '<p>' + '\n'.join(output_lines) + '</p>', recovered_tokens, true_predictions, recovered_tokens_asr, true_predictions_asr
    else:
        raise NotImplementedError()


text = st.text_input('Input Text', 'Life of Brian')
res, tok, pred = predict_text(text, 'txt')
msk_txt, org_phr, tok, pred = make_postprocessed_tokens_2(tok, pred)
components.html(prettify_html(msk_txt))
print('TP', tok, pred)
fixed = apply_fixations_for_single_text(models_package, tok[0], pred[0])
components.html(prettify_html(fixed))

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    # bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    string_data = stringio.read()
    # result, tokens, predictions = predict_text(string_data, 'jsonl')
    result, tokens, predictions, tokens_asr, predictions_asr = predict_text(string_data, 'jsonl')

    # masked_text, original_phrases = make_postprocessed_tokens_2(tokens, predictions)
    masked_text, original_phrases = make_postprocessed_tokens_with_asr_signal(tokens, predictions, tokens_asr, predictions_asr)

    components.html(prettify_html(masked_text), scrolling=True, height=600)

    fixed_html = apply_fixations(models_package, 'runtime_results/input.json')

    components.html(prettify_html(fixed_html), scrolling=True, height=600)
