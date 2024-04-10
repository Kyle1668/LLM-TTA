import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

adaptive_model_name = "stabilityai/StableBeluga-7B"
adaptive_tokenizer = AutoTokenizer.from_pretrained(adaptive_model_name, trust_remote_code=True)
adaptive_model = AutoModelForCausalLM.from_pretrained(adaptive_model_name, device_map="auto").eval()


prompt_input = "Every box of PLA from Hatchbox has been so nice and worry free for printing."
prompt = """
### Instructions ###
The assistant is to paraphrase the input text.

### Input Text ###
Now paraphrase {{{"<style_input>"}}}.

Return the text in the format: {{{Paraphased Text}}}.

### Paraphased Text ###
Paraphased Input Text: {{{### Assistant:
""".replace("<style_input>", prompt_input)
print(prompt)

tokenized_prompt = adaptive_tokenizer.encode(prompt, return_tensors="pt").to(adaptive_model.device)
outputs = adaptive_model.generate(
                tokenized_prompt,
                temperature=0,
                max_new_tokens=1000,
                early_stopping=True,
                return_dict_in_generate=True,
                num_return_sequences=4,
                num_beam_groups=4,
                num_beams=4,
                diversity_penalty=0.5,
            )

print(outputs)
decoded_text = adaptive_model.batch_decode(outputs["sequences"])
print(decoded_text)