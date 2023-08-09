import os
import time
import torch
import openai
from huggingface_hub import hf_hub_download
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding,
)
from transformers.modeling_outputs import CausalLMOutput

class OpenAIModelConfig(PretrainedConfig):
    def __init__(self, name_or_path):
        self.name_or_path = name_or_path
        self.architectures = ["AutoModelForCausalLM"]
        self.openai_api_key = os.environ["OPENAI_API_KEY"]

class OpenAIModel(PreTrainedModel):
    def __init__(self, config: OpenAIModelConfig):
        super().__init__(config)
        self.name_or_path = config.name_or_path
        self.config = config

    def generate(
        self,
        prompt,
        do_sample=False,
        temperature=0.0,
        max_new_tokens=100,
        early_stopping=True,
        return_dict_in_generate=True,
        # Dummy arguments
        **kwargs
    ):
        messages = self.parse_messages(prompt)
        return self.send_messages(do_sample, temperature, max_new_tokens, messages)

    def send_messages(self, do_sample, temperature, max_new_tokens, messages):
        try:
            if self.name_or_path == "gpt-4":
                time.sleep(5)

            completion = openai.ChatCompletion.create(
                model=self.name_or_path,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature if do_sample else 0.0)

            return completion
        except Exception as error:
            print(error)
            time.sleep(10)
            return self.send_messages(do_sample, temperature, max_new_tokens, messages)

    def parse_messages(self, prompt):
        is_rewrite_task = "### Input Text ###" in prompt
        if is_rewrite_task:
            system_message = prompt.split("### Input Text ###")[0]
            user_message = "### Input Text ###" + prompt.split("### Input Text ###")[1]
            return [
                    {"role": "system", "content" : system_message},
                    {"role": "user", "content" : user_message}
                ]

        delimiter = "What is the label for the following text? You must decide which label the text is."
        system_message = prompt.split(delimiter)[0]
        user_message = delimiter + prompt.split(delimiter)[1]
        return [
                    {"role": "system", "content" : system_message},
                    {"role": "user", "content" : user_message}
                ]