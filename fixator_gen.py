from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig, AutoModelForSeq2SeqLM
from transformers import AutoConfig
import torch, json
from munch import Munch
from postprocess import make_sample


def load_generator(model_name, tokenizer_name, config_name, device = torch.device('cuda')):
    config = BartConfig.from_pretrained(config_name, force_bos_token_to_be_generated=True)
    model = BartForConditionalGeneration.from_pretrained(model_name, config=config).eval().to(device)
    tok = BartTokenizer.from_pretrained(tokenizer_name)
    # client_specific_words = list(config['client_specific_words'])
    return Munch(
        model=model,
        config=config,
        tokenizer=tok,
        device=device,
        # client_specific_words=client_specific_words
    )





def highlight_fixations(sample, generated_output=None):
    # print('S', sample)
    # print('G', generated_output)
    chunks = []
    N = len(sample['target'])
    source_text = sample['source']
    target_text = generated_output or sample['target']
    if target_text.count('#') != source_text.count('#'):
        print('SOURCE: {} \nTARGET: {}'.format(source_text, target_text))
        sample['highlighted_result'] = target_text
        return sample
    while True:
        if '#' not in target_text:
            chunks.append(target_text)
            break
        idx1 = target_text.index('#')
        chk = target_text[:idx1]
        chunks.append(chk)
        target_text = target_text[idx1+1:]
        idx2 = target_text.index('#')
        generated_text = target_text[:idx2]
        generated_text = '<strong>{}</strong>'.format(generated_text)
        chunks.append(generated_text)
        target_text = target_text[idx2+1:]
    

    
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
    # sample['highlighted_result'] = generated_output or sample['target']
    sample['highlighted_result'] = ' '.join(chunks)
    return sample


def apply_fixations(model_package, source_file=None, samples=None, context_gap=2):
    import json

    assert (samples != source_file), 'Either source file or samples must be specified'

    if samples is None:
        with open(source_file) as fin:
            samples = json.load(fin)
    
    resulting_samples = []
    N = len(samples)
    i = 0
    cache = []
    while i < N:
        s = samples[i]
        if '#' not in s['source']:
            cache.append(s)

            # resulting_samples.append(highlight_fixations(s))
            i += 1
            continue
        else:
            cache = list(reversed(cache))
            left_context = []
            for l in range(min(len(cache), context_gap)):
                left_context.append(cache[l]['source'].strip().rstrip('$%'))
            cache = cache[min(len(cache), context_gap):]
            for c in reversed(cache):
                resulting_samples.append(highlight_fixations(c))
            
            cache = []
            
            left_context = list(reversed(left_context))

            right_context = []
            r = i + 1
            for r in range(i + 1, min(N, i + context_gap + 1)):
                if '#' not in samples[r]['source']:
                    right_context.append(samples[r]['source'].strip().rstrip('$%'))
                else:
                    break
            
            i = r

            left_context = ' '.join(left_context)
            right_context = ' '.join(right_context)

            suffix = s['source'][s['source'].index('$%'):]
            cleaned_text = s['source'][:s['source'].index('$%')].strip()

            example_english_phrase = ' '.join([left_context, cleaned_text, right_context, suffix])
        
        # example_english_phrase = s['source']

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
    
    for c in cache:
        resulting_samples.append(highlight_fixations(c))

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
