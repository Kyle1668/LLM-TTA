import os
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

        try:
            messages = self.parse_messages(prompt)

            completion = openai.ChatCompletion.create(
                model=self.name_or_path,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature if do_sample else 0.0)

            return completion
        except Exception as error:
            print(error)
            raise error

    def parse_messages(self, prompt):
        system_message = prompt.split("### Input Text ###")[0]
        user_message = "### Input Text ###" + prompt.split("### Input Text ###")[1]
        return [
                {"role": "system", "content" : system_message},
                {"role": "user", "content" : user_message}
            ]
