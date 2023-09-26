from huggingface_hub import hf_hub_download
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding,
)
from transformers.modeling_outputs import CausalLMOutput
import nlpaug.augmenter.word as naw

class AugmenterConfig(PretrainedConfig):
    def __init__(self, name_or_path, action):
        self.name_or_path = name_or_path
        # self.architectures = ["AutoModelForCausalLM"]
        self.action = action

class AugmenterModel(PreTrainedModel):
    def __init__(self, config: AugmenterConfig):
        super().__init__(config)
        self.name_or_path = config.name_or_path
        self.config = config
        self.augmenter = naw.ContextualWordEmbsAug(action=config.action, device="cuda")

    def generate(
        self,
        prompt_batch,
        do_sample=False,
        temperature=0.0,
        max_new_tokens=100,
        early_stopping=True,
        return_dict_in_generate=True,
        # Dummy arguments
        **kwargs
    ):
        if isinstance(prompt_batch, str):
            return [self.augmenter.augment(prompt_batch, n=4)]

        return [self.augmenter.augment(prompt, n=4) for prompt in prompt_batch]
