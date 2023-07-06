from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import torch


def is_large_language_model(model_name):
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return model_config.architectures[0].endswith("ForCausalLM")


def is_language_model(model_name):
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    is_seq2seq_lm = model_config.architectures[0].endswith("ForConditionalGeneration")
    return is_seq2seq_lm or is_large_language_model(model_name)


def get_model_objects(model_name, num_labels, training=False):
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    is_seq2seq_lm = model_config.architectures[0].endswith("ForConditionalGeneration")
    is_qa_model = model_config.architectures[0].endswith("ForQuestionAnswering")
    is_llm = model_config.architectures[0].endswith("ForCausalLM")
    is_llama_based_model = is_llm and "llama" in model_name or "vicuna" in model_name

    tokenizer = LlamaTokenizer.from_pretrained(model_name) if is_llama_based_model else AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<s>" if tokenizer.pad_token in [None, ""] and str(tokenizer.eos_token) in [None, ""] else tokenizer.eos_token

    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    numerical_precision = torch.float32 if training else torch.float16
    if is_llm:
        num_billions = [int(entry[:-1]) for entry in model_name.split("-") if entry[0].isdigit() and entry.lower().endswith("b")]
        load_in_8bit = (len(num_billions) > 0 and num_billions[0] > 7) or training
        if load_in_8bit:
            print("Loading in 8-bit mode since the model has more than 7B parameters or we are training.")
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, load_in_8bit=True, llm_int8_threshold=0, device_map="auto").eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=numerical_precision).eval().to(device)
    elif is_qa_model:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, trust_remote_code=True, torch_dtype=numerical_precision).eval().to(device)
    elif is_seq2seq_lm:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=numerical_precision).eval().to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, num_labels=num_labels).eval().to(device)
    return tokenizer, model
