from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, FalconForCausalLM, FalconConfig
from accelerate import infer_auto_device_map
import torch
import torch.distributed as dist
from openai_hf.openai_model import OpenAIModel, OpenAIModelConfig
from augmenters_hf.augmenter import AugmenterModel, AugmenterConfig


model_config = {}

def select_device():
    if torch.cuda.is_available():
        print("Using CUDA.")
        return "cuda"
    if torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("Warning: Torch MPS is not built. Falling back to CPU.")
        
        print("Using MPS.")
        return "mps"
    
    print("Using CPU.")
    return "cpu"


def is_large_language_model(model_name):
    global model_config

    if model_name.startswith("aug"):
        return False

    if is_openai_model(model_name):
        return True

    if model_name not in model_config:
        if "falcon" in model_name:
            model_config[model_name] = FalconConfig.from_pretrained(model_name)
        else:
            model_config[model_name] = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    return model_config[model_name].architectures[0].endswith("ForCausalLM")


def is_language_model(model_name):
    global model_config

    if model_name.startswith("aug"):
        return False

    if is_openai_model(model_name):
        return True

    if model_name not in model_config:
        if "falcon" in model_name:
            model_config[model_name] = FalconConfig.from_pretrained(model_name)
        else:
            model_config[model_name] = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    is_seq2seq_lm = model_config[model_name].architectures[0].endswith("ForConditionalGeneration")
    return is_seq2seq_lm or is_large_language_model(model_name)


def is_openai_model(model_name):
    return model_name in ["gpt-3.5-turbo", "gpt-4"]


def get_model_objects(model_name, num_labels, training=False):
    if dist.is_initialized():
        # Load the model on rank one first. This avoids race conditions if the model must be downloaded
        # from the HuggingFace model hub at the expense of extra time.
        if dist.get_rank() == 0:
            model = get_model(model_name, num_labels, training=training)
            print(f"{model_name} loaded on rank {dist.get_rank()}.")

        # Other ranks will wait for rank zero to catch up before loading the model.
        dist.barrier()
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()} finished loading {model_name}. Now loading on the remaining {dist.get_world_size() - 1} ranks.")

        # Load the model on the remaining ranks now that rank zero has finished.
        if dist.get_rank() != 0:
            model = get_model(model_name, num_labels, training=training)
            print(f"{model_name} loaded on rank {dist.get_rank()}.")

        return model

    return get_model(model_name, num_labels, training=training)


def get_model(model_name, num_labels, training=False):
    if model_name.startswith("aug"):
        action = model_name.split("_")[1]
        return None, AugmenterModel(AugmenterConfig(model_name, action))
    if is_openai_model(model_name):
        return None, OpenAIModel(OpenAIModelConfig(model_name))

    is_llama_based_model = "llama" in model_name or "vicuna" in model_name
    is_falcon_based_model = "falcon" in model_name
    model_config = None
    if is_falcon_based_model:
        model_config = FalconConfig.from_pretrained(model_name)
    else:
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    is_seq2seq_lm = model_config.architectures[0].endswith("ForConditionalGeneration")
    is_qa_model = model_config.architectures[0].endswith("ForQuestionAnswering")
    is_llm = model_config.architectures[0].endswith("ForCausalLM")
    is_embedding_model = model_name in ["princeton-nlp/sup-simcse-roberta-large"]

    tokenizer = None
    if is_llama_based_model:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<s>" if tokenizer.pad_token in [None, ""] and str(tokenizer.eos_token) in [None, ""] else tokenizer.eos_token

    model = None
    device = select_device()
    numerical_precision = torch.float32 if training else torch.float16
    if is_llm:
        num_billions = [float(entry[:-1]) for entry in model_name.split("-") if entry[0].isdigit() and entry.lower().endswith("b")]
        large_models = ["stabilityai/StableBeluga2"]
        needs_quantization = (len(num_billions) > 0 and num_billions[0] >= 13) or training or model_name in large_models
        load_in_8bit = needs_quantization and torch.cuda.get_device_capability()[0] >= 8
        load_in_4bit = needs_quantization and torch.cuda.get_device_capability()[0] < 8
        if device == "cpu":
            print("Loading LLM on CPU since no GPU is available.")
            if is_falcon_based_model:
                model = FalconForCausalLM.from_pretrained(model_name, torch_dtype=numerical_precision).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=numerical_precision).eval()
        elif load_in_4bit:
            print("Loading in 4-bit mode since the model has more than 7B parameters and 8bit isn't supported.")
            if is_falcon_based_model:
                model = FalconForCausalLM.from_pretrained(model_name, load_in_4bit=True).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_4bit=True).eval()
        elif load_in_8bit and not dist.is_initialized():
            print("Loading in 8-bit mode since the model has more than 13B parameters or we are training.")
            if is_falcon_based_model:
                model = FalconForCausalLM.from_pretrained(model_name, load_in_8bit=True, llm_int8_threshold=0, device_map="auto").eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_8bit=True, llm_int8_threshold=0, device_map="auto").eval()
        elif dist.is_initialized() or device == "mps":
            if is_falcon_based_model:
                model = FalconForCausalLM.from_pretrained(model_name, torch_dtype=numerical_precision).eval().to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=numerical_precision).eval().to(device)
        else:
            if is_falcon_based_model:
                model = FalconForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto").eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_8bit=True, device_map="auto").eval()

    elif is_qa_model:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, trust_remote_code=True, torch_dtype=numerical_precision).eval().to(device)
    elif is_seq2seq_lm:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=numerical_precision).eval().to(device)
    elif is_embedding_model:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=numerical_precision).eval().to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, num_labels=num_labels).eval().to(device)

    # if is_falcon_based_model:
    #     model.to_bettertransformer()

    return tokenizer, model
