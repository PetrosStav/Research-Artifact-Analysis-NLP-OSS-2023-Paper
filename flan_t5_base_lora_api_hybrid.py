import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from peft import PeftModel, PeftModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", cache_dir="./models_cache", torch_dtype="auto", device_map="auto")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", cache_dir="./models_cache")

model = PeftModel.from_pretrained(model, "./model_checkpoints/sweep_hybrid/flan_t5_base_lora_LORA_SEQ_2_SEQ_LM_smooth-sweep-1_16_16_0.4_4", device_map={'': 0})

print(f"Running merge_and_unload")
model = model.merge_and_unload()

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
    import timeit

    def time_infer_text():
        infer_text('"### Snippet:\nThe training of our deep learning model was performed using the PyTorch framework (version 1.9.0), with the CIFAR-10 dataset. The framework can be downloaded from https://pytorch.org.\n\n### Question:\nList all the artifacts in the above snippet.\n\n### Answer:\n"', max_new_tokens=5)

    k = 20
    print(f"Average time taken for {k} runs of infer_text():", timeit.timeit(time_infer_text, number=k) / k)
