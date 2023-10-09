import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", cache_dir="/storage4/big_transformer_models/flan-t5-xl", torch_dtype="auto", device_map="auto")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", cache_dir="/storage4/big_transformer_models/flan-t5-xl")


print(model.hf_device_map)
def infer_text_web(sentence, max_len, **kwargs):
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(model.device)
    
    if max_len == 0:
        outputs = model.generate(input_ids=inputs, **kwargs)
    else:
        outputs = model.generate(input_ids=inputs, max_new_tokens=max_len, **kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def infer_text(sentence, **kwargs):
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(model.device)

    outputs = model.generate(input_ids=inputs, **kwargs)
    if 'return_dict_in_generate' in kwargs:
        return outputs
    return [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]


def tokenize_text(sentence):
    return tokenizer.encode(sentence, add_special_tokens=False)


if __name__ == '__main__':
    res = infer_text('"### Snippet:\nThe training of our deep learning model was performed using the PyTorch framework (version 1.9.0), with the CIFAR-10 dataset. The framework can be downloaded from https://pytorch.org.\n\n### Question:\nList all the artifacts in the above snippet.\n\n### Answer:\n"')
    print()
