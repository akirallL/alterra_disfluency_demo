from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig, AutoModelForSeq2SeqLM
from transformers import AutoConfig
import torch, json
from munch import Munch
from postprocess import make_sample


def load_generator(model_name, tokenizer_name, config_name, device = torch.device('cuda')):
    config = BartConfig.from_pretrained(config_name, force_bos_token_to_be_generated=True)
    model = BartForConditionalGeneration.from_pretrained(model_name, config=config).eval().to(device)
    tok = BartTokenizer.from_pretrained(tokenizer_name)
    return Munch(
        model=model,
        config=config,
        tokenizer=tok,
        device=device
    )

def highlight_fixations(sample, generated_output=None):
    # # print('S', sample)
    # # print('G', generated_output)
    # chunks = []
    # predictions = []
    # total_len = 0
    # N = len(sample['target'])
    # l_ptr = 0
    # remaining_text = sample['source']
    # target_text = generated_output or sample['target']
    # tgt_ptr = 0
    # while True:
    #     if '#' not in remaining_text:
    #         chunks.append(remaining_text)
    #         break
    #     idx1 = remaining_text.index('#')
    #     chk = remaining_text[:idx1]
    #     chunks.append(chk)
    #     remaining_text = remaining_text[idx1+1:]
    #     idx2 = remaining_text.index('#')
    #     remaining_text = remaining_text[idx2+1:]
    
    # # print('C', chunks)

    # assert target_text.startswith(chunks[0]), '{} {}'.format(chunks[0], target_text)
    # target_text = target_text[len(chunks[0]):]

    # for chk in chunks[1:]:
    #     assert chk in target_text, '{} {}'.format(chk, target_text)
    #     idx = target_text.index(chk)
    #     pred = target_text[:pred]
    #     target_text = target_text[idx + len(chk):]
    #     predictions.append(pred)

    # resulting_text = chunks[0]
    # for i in range(len(predictions)):
    #     resulting_text.append('<strong>{}</strong>'.format(predictions[i]))
    #     resulting_text.append(chunks[i + 1])
    # resulting_text = ' '.join(resulting_text)

    # sample['highlighted_result'] = resulting_text
    sample['highlighted_result'] = generated_output or sample['target']
    return sample



def apply_fixations(model_package, source_file=None, samples=None):
    import json

    assert (samples != source_file), 'Either source file or samples must be specified'

    if samples is None:
        with open(source_file) as fin:
            samples = json.load(fin)
    
    resulting_samples = []
    for s in samples:
        if '#' not in s['source']:
            resulting_samples.append(highlight_fixations(s))
            continue
        example_english_phrase = s['source']
        print('IN: ', example_english_phrase)

        batch = model_package.tokenizer(example_english_phrase, return_tensors='pt')
        generated_ids = model_package.model.generate(
            batch['input_ids'].to(model_package.device),
            max_length=128, 
            num_beams=10, 
            num_return_sequences=10
        )
        sent = model_package.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print('OUT:', sent)
        resulting_samples.append(highlight_fixations(s, sent))
    return '<br>'.join([s['highlighted_result'] for s in resulting_samples])


def apply_fixations_for_single_text(model_package, tokens, labels):
    # tokens = text.split()
    # labels = [1 for _ in tokens]
    sample = make_sample(tokens, labels)
    result = apply_fixations(model_package, samples=[sample])
    print(sample, tokens, labels)
    return result


with open('config.json') as fl:
    config = json.load(fl)

models_package = \
    load_generator(device=torch.device('cuda'), **config['generator_model'])
