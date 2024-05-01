from huggingface_hub import hf_hub_download
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding,
    pipeline,
)
from transformers.modeling_outputs import CausalLMOutput
import nlpaug.augmenter.word as naw
import torch

class AugmenterConfig(PretrainedConfig):
    def __init__(self, name_or_path, action):
        self.name_or_path = name_or_path
        self.action = action.replace("-", "_")

class AugmenterModel(PreTrainedModel):
    def __init__(self, config: AugmenterConfig):
        super().__init__(config)
        self.name_or_path = config.name_or_path
        self.config = config
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if config.action == "back_translate":
            self.en_de_translator = pipeline("translation", model="facebook/wmt19-en-de", device=device)
            self.de_en_translator = pipeline("translation", model="facebook/wmt19-de-en", device=device)
        else:
            self.augmenter = naw.ContextualWordEmbsAug(action=config.action, device=device)

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
        num_agumentations = 4
        augmentations = []
        for prompt in (prompt_batch if isinstance(prompt_batch, list) else [prompt_batch]):
            if self.config.action == "back_translate":
                german_version = self.en_de_translator(prompt)[0]["translation_text"]
                # english_versions = self.de_en_translator(
                #     german_version,
                #     num_return_sequences=4,
                #     temperature=0.7,
                #     num_beams=4,
                #     num_beam_groups=4,
                #     top_p=0.95,
                #     top_k=0,
                #     repetition_penalty=10.0,
                #     diversity_penalty=1.0,
                #     no_repeat_ngram_size=2)

                english_versions = self.de_en_translator(german_version, num_return_sequences=4, temperature=0.3)
                augmentations.append([translation["translation_text"] for translation in english_versions])
            else:
                augmentations.append(self.augmenter.augment(prompt, n=num_agumentations))

        return augmentations
