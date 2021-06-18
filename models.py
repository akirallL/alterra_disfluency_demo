from disfluency_detector import STOP_WORDS
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertConfig, BertModel, DistilBertPreTrainedModel, DistilBertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn import TransformerEncoderLayer
from munch import Munch
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers import DistilBertTokenizerFast, BertTokenizerFast, DataCollatorForTokenClassification
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
from copy import deepcopy
import streamlit as st
from phonemes import get_top_phoneme_neighbors


class FloatTransformer(nn.Module):
    def __init__(self, hidden_dim=8, nhead=4, drop_p=0.1):
        super().__init__()
        self.embedding = nn.Linear(1, hidden_dim)
        self.dropout = nn.Dropout(p=drop_p)
        self.transformer = nn.TransformerEncoderLayer(d_model=8, nhead=4)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.transformer(x)
        return x


def copy_classifier_weights(clf1, clf2):
#     n, m = clf2.weight.size(0), clf2.weight.size(1)
#     clf1.weight[:n, :m] = clf2.weight
#     clf1.bias[:] = clf2.bias

    n, m = clf2.weight.size(0), clf2.weight.size(1)
    print(clf2.weight.shape, clf1.weight[:, m:].shape)
    weight = torch.cat([clf2.weight, clf1.weight[:, m:]], dim=1)
    clf1.weight.data = weight
    clf1.bias.data = clf2.bias.data
    return clf1, clf2


class DistilBertForTokenClassWithConfidences(DistilBertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, use_confidences=True):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.use_confidences = use_confidences
        if self.use_confidences:
            float_emb_dim = 8
            self.classifier = nn.Linear(config.hidden_size + float_emb_dim, config.num_labels)
            self.float_embedder = FloatTransformer(float_emb_dim, 4, 0.2)
#             self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        confidences=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.use_confidences:
            confidences = confidences.unsqueeze(2)
            confidences = self.float_embedder(confidences)
            
            sequence_output = torch.cat([sequence_output, confidences], dim=2)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@st.cache
def load_model(model_path):
    config = BertConfig.from_pretrained('distilbert-base-uncased')
    config.num_labels = 2
    model = DistilBertForTokenClassWithConfidences(config, use_confidences=True)
    model.load_state_dict(torch.load(model_path))
    model.eval().cuda()
    return model



# @st.cache
def load_highlighter(model_path):
    tok = BertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # return Munch(tok=tok)
    model = load_model(model_path)
    return Munch(model=model, tok=tok)


def tokenize_and_align_labels(examples, 
                              tokenizer, 
                              max_num=None, 
                              tokenizer_batch_len=None, 
                              label_all_tokens = True):
    if max_num is None:
        max_num = len(examples['tokens'])
    else:
        max_num = min(max_num, len(examples['tokens']))
        
    all_word_ids = []
        
    if tokenizer_batch_len is not None and max_num > tokenizer_batch_len:
        tokenized_inputs = tokenizer(examples["tokens"][:tokenizer_batch_len], 
                                     truncation=True, 
                                     is_split_into_words=True)
        
        for j in range(len(tokenized_inputs['input_ids'])):
            all_word_ids.append(tokenized_inputs.word_ids(batch_index=j))
        
        for i in tqdm(range(tokenizer_batch_len, max_num, tokenizer_batch_len)):
            max_idx = min(max_num, i + tokenizer_batch_len)
            sample = examples["tokens"][i:max_idx]
            tokenized_inputs_local = \
                tokenizer(sample, truncation=True, is_split_into_words=True)
            for k in ['input_ids', 'attention_mask',]:
                tokenized_inputs[k].extend(tokenized_inputs_local[k])
            for j in range(len(tokenized_inputs_local['input_ids'])):
                all_word_ids.append(tokenized_inputs_local.word_ids(batch_index=j))
    else:
        tokenized_inputs = tokenizer(examples["tokens"][:max_num], truncation=True, is_split_into_words=True)
        for i in range(max_num):
            all_word_ids.append(tokenized_inputs.word_ids(batch_index=i))

    labels = []
    confidences = []
    for i, word_confidences in enumerate(tqdm(examples["confidences"][:max_num], total=max_num)):
        label = [0 for _ in range(len(word_confidences))]
        word_ids = all_word_ids[i]
        previous_word_idx = None
        label_ids = []
        token_confidences = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
                token_confidences.append(0)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
                token_confidences.append(word_confidences[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
                token_confidences.append(word_confidences[word_idx] if label_all_tokens else 0)
            previous_word_idx = word_idx

        labels.append(label_ids)
        confidences.append(token_confidences)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["confidences"] = confidences
    tokenized_inputs['words'] = examples['tokens']
    tokenized_inputs['initial_confidences'] = examples['confidences']
    return tokenized_inputs


class SentenceWithConfidencesAndContext:
    def __init__(self, main_context, left_context, right_context):
        self.main_context = main_context
        self.left_context = left_context
        self.right_context = right_context
        self.merge_contexts()

    def merge_contexts(self):
        tokenization_result = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'labels': [], 'confidences': []}

        self.left_border = len(self.left_context['input_ids'])
        self.right_border = self.left_border + len(self.main_context['input_ids'])

        for i, tokenizer_outputs in enumerate([self.left_context, self.main_context, self.right_context]):
            if i != 1:
                for k, v in tokenization_result.items():
                    v.extend(tokenizer_outputs[k])
            else:
                for k, v in tokenization_result.items():
                    v.extend(tokenizer_outputs[k][1:-1])

        self.tokenization_result = tokenization_result
        return self.tokenization_result

    def match_result_with_predictions(self, predictions):
        self.true_tokens = self.main_context['input_ids'][1:-1] #self.tokenization_result['input_ids'][self.left_border:self.right_border]
        self.true_predictions = predictions[self.left_border:self.right_border]
        return self.true_tokens, self.true_predictions


def make_contexted_samples(tokenized_inputs, sentences_from_sides_gap=3, max_input_length=256):
    def accumulate_context(tok_inputs, l, r, is_left):
        input_ids = []
        confidences = []
        labels = []
        token_type_ids = []
        attention_mask = []
        if is_left:
            input_ids.append(tok_inputs['input_ids'][l][0])
            confidences.append(tok_inputs['confidences'][l][0])
            labels.append(tok_inputs['labels'][l][0])
            token_type_ids.append(tok_inputs['token_type_ids'][l][0])
            attention_mask.append(tok_inputs['attention_mask'][l][0])
        
        for i in range(l, r):
            input_ids.extend(tok_inputs['input_ids'][i][1:-1])
            confidences.extend(tok_inputs['confidences'][i][1:-1])
            labels.extend(tok_inputs['labels'][i][1:-1])
            token_type_ids.extend(tok_inputs['token_type_ids'][i][1:-1])
            attention_mask.extend(tok_inputs['attention_mask'][i][1:-1])

        if not is_left:
            input_ids.append(tok_inputs['input_ids'][r - 1][-1])
            confidences.append(tok_inputs['confidences'][r - 1][-1])
            labels.append(tok_inputs['labels'][r - 1][-1])
            token_type_ids.append(tok_inputs['token_type_ids'][r - 1][-1])
            attention_mask.append(tok_inputs['attention_mask'][r - 1][-1])
        
        return {
            'input_ids': input_ids,
             'confidences': confidences,
              'labels': labels, 
              'attention_mask': attention_mask,
               'token_type_ids': token_type_ids
               }

    def expand_inputs(tok_inputs, l, r):
        N = len(tok_inputs['input_ids'])
        total_len = sum([len(tok_inputs['input_ids'][i]) for i in range(l, r)])
        ll, rr = l, r
        i = 0
        able_to_move_left, able_to_move_right = True, True
        while ll >= 0 and rr <= N and total_len <= max_input_length and (able_to_move_left or able_to_move_right):
            # print(i, ll, rr, N, able_to_move_left, able_to_move_right, total_len)
            if i % 2 == 0:
                pre_ll = ll
                if ll == 0:
                    able_to_move_left = False
                else:
                    ll -= 1
                # ll = max(0, ll - 1)
                total_len = sum([len(tok_inputs['input_ids'][i]) for i in range(ll, rr)])
                if total_len > max_input_length:
                    ll = pre_ll
                    able_to_move_left = False
            else:
                pre_rr = rr
                if rr == N:
                    able_to_move_right = False
                else:
                    rr += 1
                # rr = min(r + 1, N)
                total_len = sum([len(tok_inputs['input_ids'][i]) for i in range(ll, rr)])
                if total_len > max_input_length:
                    rr = pre_rr
                    able_to_move_right = False
            # total_len = sum([len(tok_inputs['input_ids'][i]) for i in range(ll, rr)])
            i += 1
        return ll, rr



    sentences_with_contexts = []
    N = len(tokenized_inputs['input_ids'])
    for i in range(N):
        # left_border = max(0, i - sentences_from_sides_gap)
        # right_border = min(N, i + sentences_from_sides_gap + 1)
        left_border, right_border = expand_inputs(tokenized_inputs, i, i + 1)
        left_context = accumulate_context(tokenized_inputs, left_border, i, is_left=True)
        right_context = accumulate_context(tokenized_inputs, i + 1, right_border, is_left=False)
        main_context = {k: tokenized_inputs[k][i] for k in tokenized_inputs.keys()}
        # print('Left_context:', left_context)
        # print('Central part:', sentences[i])
        # print('RightContext:', right_context)
        # print('\n\n')
        sentences_with_contexts.append(SentenceWithConfidencesAndContext(main_context, left_context, right_context))
    return sentences_with_contexts


def pad_seq(seq, max_length):
    while len(seq) < max_length:
        seq.append(0)
    return seq


class TensorDataset(Dataset):
    def __init__(self, samples, tok, max_length=128, use_token_type_ids=False):
        super().__init__()
        self.tok = tok
        self.data_collator = DataCollatorForTokenClassification(tok, max_length=max_length, padding='max_length')
        raw_samples = [s.tokenization_result for s in samples]
        for s in raw_samples:
            while len(s['confidences']) < max_length:
                s['confidences'].append(0)
        self.confidences = torch.FloatTensor([s['confidences'] for s in raw_samples])
        print('CONFIDENCES INFO', self.confidences.sum(), 
              (self.confidences != 0).float().sum() / self.confidences.nelement(), self.confidences.shape)
        self.samples = deepcopy(raw_samples)
        for s in self.samples:
            del s['confidences']
        self.collated = self.data_collator(self.samples)
        self.use_token_type_ids = use_token_type_ids

    def __getitem__(self, idx):
        result = {k: self.collated[k][idx] for k in self.collated if k != 'token_type_ids' or \
                                                                    self.use_token_type_ids}
        result['confidences'] = self.confidences[idx]
        return result

    def __len__(self):
        return len(self.collated['input_ids'])


def glue_tokens(tokens, predictions):
    new_tokens, new_labels = [], []
    # print(tokens)
    for token, label_idx in zip(tokens, predictions):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
            new_labels[-1] = max(new_labels[-1], label_idx)
        else:
            new_labels.append(label_idx)
            new_tokens.append(token)
    # print(new_tokens)
    return new_tokens, new_labels


def _clip_stopwords(tokens, stop_words):
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


def make_highlighted_string(tokens, indicators, CLIENT_WORDS=None):
    THRESCHOLD = 0.5

    from stop_words import get_stop_words
    STOP_WORDS = get_stop_words('en')
    l, r = 0, 0
    N = len(tokens)
    new_tokens = []
    original_phrases = []
    while l < N:
        if indicators[l] <= THRESCHOLD:
            new_tokens.append(tokens[l])
            l += 1
        else:
            r = l
            while r < N and indicators[r] > THRESCHOLD or tokens[max(0, r - 1)] == '\'':
                r += 1
            is_all_stopwords, pref, med, suf = _clip_stopwords(tokens[l:r], STOP_WORDS)
            brightness = max(indicators[l:r] + [0])
            brightness = int(brightness * 10) * 10
            # print(brightness)

            if brightness >= THRESCHOLD and CLIENT_WORDS is not None:
                candidates = get_top_phoneme_neighbors(med, CLIENT_WORDS)
                tooltip = 'data-tooltip="{}"'.format(' '.join(candidates))
            else:
                tooltip = ''

            if not is_all_stopwords:
                new_tokens.append('{} <strong class="brightness-{}" {}>{}</strong> {}'.format(pref, brightness, tooltip, med, suf))
            else:
                new_tokens.append('{} {} {}'.format(pref, med, suf))
            original_phrases.append(tokens[l:r])
            l = r
    text = ' '.join(new_tokens)
    return text


def make_highlighted_text(tokens_list, predictions_list, CLIENT_WORDS=None):
    def prettify_html(html):
        pref = '''
        <style>
        .brightness-0 {
            background-color: rgba(0,255,255,0.0);
            opacity: 0.0;
        }
        .brightness-10 {
            background-color: rgba(0,255,255,0.1);
            opacity: 0.1;
        }
        .brightness-20 {
            background-color: rgba(0,255,255,0.2);
            opacity: 0.2;
        }
        .brightness-30 {
            background-color: rgba(0,255,255,0.3);
            opacity: 0.3;
        }
        .brightness-40 {
            background-color: rgba(0,255,255,0.4);
            opacity: 0.4;
        }
        .brightness-50 {
            background-color: rgba(0,255,255,0.5);
            opacity: 0.5;
        }
        .brightness-60 {
            background-color: rgba(0,255,255,0.6);
            opacity: 0.6;
        }
        .brightness-70 {
            background-color: rgba(0,255,255,0.7);
            opacity: 0.7;
        }
        .brightness-80 {
            background-color: rgba(0,255,255,0.8);
            opacity: 0.8;
        }
        .brightness-90 {
            background-color: rgba(0,255,255,0.9);
            opacity: 0.9;
        }
        .brightness-100 {
            background-color: rgba(0,255,255,1.0);
            opacity: 1.0;
        }
        [data-tooltip] {
            position: relative; /* Относительное позиционирование */ 
        }
        [data-tooltip]::after {
            content: attr(data-tooltip); /* Выводим текст */
            position: absolute; /* Абсолютное позиционирование */
            width: 300px; /* Ширина подсказки */
            left: 0; top: 0; /* Положение подсказки */
            background: #3989c9; /* Синий цвет фона */
            color: #fff; /* Цвет текста */
            padding: 0.5em; /* Поля вокруг текста */
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3); /* Параметры тени */
            pointer-events: none; /* Подсказка */
            opacity: 0; /* Подсказка невидима */
            transition: 1s; /* Время появления подсказки */
        } 
        [data-tooltip]:hover::after {
            opacity: 1; /* Показываем подсказку */
            top: 2em; /* Положение подсказки */
        }
        </style>
        <p style="color: green;">
        '''
        suf = '\n</p>'
        return pref + html + suf
    lines = [make_highlighted_string(t, p, CLIENT_WORDS=CLIENT_WORDS) for t, p in zip(tokens_list, predictions_list)]
    text = '<br>\n'.join(lines)
    return prettify_html(text)

