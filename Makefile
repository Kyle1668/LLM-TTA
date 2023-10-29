########## Configure Environment ##########

create_env:
	conda create -n eval-aug python=3.10

install_depends:
	pip install -r requirements.txt
	mkdir datasets; cd datasets && wget https://huggingface.co/datasets/Kyle1668/BOSS-Robustness-Benchmark/resolve/main/BOSS.zip && unzip BOSS.zip
	mv datasets/process datasets/boss_benchmark

clear_rewrites_cache:
	rm -rf cached_rewrites/*

########## Primary Experiment Sweeps ##########

main_results_id_sentiment:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=validation --evaluate_id_adaptation --model=Kyle1668/boss-sentiment-bert-base-uncased,Kyle1668/boss-sentiment-t5-large,tiiuae/falcon-7b-instruct --adaptive_model=aug_insert,aug_substitute,aug_back-translate,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

main_results_id_toxicity:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=validation --evaluate_id_adaptation --model=Kyle1668/boss-toxicity-bert-base-uncased,Kyle1668/boss-toxicity-t5-large,tiiuae/falcon-7b-instruct --adaptive_model=aug_insert,aug_substitute,aug_back-translate,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

main_results_id_news:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=sst5 --model=Kyle1668/boss-sentiment-bert-base-uncased --adaptive_model=NousResearch/Llama-2-7b-chat-hf,NousResearch/Llama-2-7b-hf --skip_style_model_eval --num_shots=0,16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

main_results_ood_sentiment:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --model=tiiuae/falcon-7b-instruct,Kyle1668/boss-sentiment-t5-large,Kyle1668/boss-sentiment-bert-base-uncased --adaptive_model=aug_insert,aug_substitute,aug_back-translate,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

main_results_ood_toxicity:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --model=tiiuae/falcon-7b-instruct,Kyle1668/boss-toxicity-t5-large,Kyle1668/boss-toxicity-bert-base-uncased --adaptive_model=aug_insert,aug_substitute,aug_back-translate,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

main_results_ood_news:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --model=tiiuae/falcon-7b-instruct,Kyle1668/ag-news-t5-large,Kyle1668/ag-news-bert-base-uncased --adaptive_model=aug_insert,aug_substitute,aug_back-translate,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

ablate_model_size:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=test --model=EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,EleutherAI/pythia-12b --adaptive_model=aug_back-translate,aug_insert,aug_substitute,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=sst5 --model=EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,EleutherAI/pythia-12b --adaptive_model=aug_back-translate,aug_insert,aug_substitute,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=toxigen --model=EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,EleutherAI/pythia-12b --adaptive_model=aug_back-translate,aug_insert,aug_substitute,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb

rewriter_model_ood_eval:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=sst5,semval,dynasent --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=toxigen,adv_civil,implicit_hate --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=test --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb

rewriter_model_eval_topk_nearest:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=sst5,semval,dynasent --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=0,16 --icl_method=topk_nearest,random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=toxigen,adv_civil,implicit_hate --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=0,16 --icl_method=topk_nearest,random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=test --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=0,16 --icl_method=topk_nearest,random --temperature=0 --trim_exemplars --use_wandb

rewriter_model_id_eval:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=validation --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=validation --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=validation --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb

########## 10/29 Reruns ##########



########## 10/23 Reruns ##########

single_node_id_sentiment_toxicity_no_paraphrase_icr:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=validation --evaluate_id_adaptation --model=Kyle1668/boss-toxicity-t5-large,Kyle1668/boss-toxicity-bert-base-uncased --adaptive_model=aug_insert,aug_substitute,aug_back-translate --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=validation --evaluate_id_adaptation --model=Kyle1668/boss-sentiment-t5-large,Kyle1668/boss-sentiment-bert-base-uncased --adaptive_model=aug_insert,aug_substitute,aug_back-translate --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	# screen -S eval_falcon_toxicity -d -m conda activate eval-aug && CUDA_VISIBLE_DEVICES=1,2,3 python evaluate_styling.py --dataset=boss_toxicity --split=validation --evaluate_id_adaptation --model=tiiuae/falcon-7b-instruct --adaptive_model=aug_insert,aug_substitute,aug_back-translate --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0
	# screen -S eval_falcon_sentiment -d -m conda activate eval-aug && CUDA_VISIBLE_DEVICES=4,5,6 python evaluate_styling.py --dataset=boss_sentiment --split=validation --evaluate_id_adaptation --model=tiiuae/falcon-7b-instruct --adaptive_model=aug_insert,aug_substitute,aug_back-translate --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars &

main_results_id_sentiment_llm:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=validation --evaluate_id_adaptation --model=tiiuae/falcon-7b-instruct,Kyle1668/boss-sentiment-t5-large,Kyle1668/boss-sentiment-bert-base-uncased --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16,0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

main_results_id_toxicity_llm:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=validation --evaluate_id_adaptation --model=tiiuae/falcon-7b-instruct,Kyle1668/boss-toxicity-t5-large,Kyle1668/boss-toxicity-bert-base-uncased --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16,0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

main_results_sentiment_tta_llama2:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --evaluate_id_adaptation --model=tiiuae/falcon-7b-instruct,Kyle1668/boss-sentiment-t5-large,Kyle1668/boss-sentiment-bert --adaptive_model=NousResearch/Llama-2-7b-chat-hf --skip_style_model_eval --num_shots=16,0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

main_results_toxicity_tta_llama2:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --evaluate_id_adaptation --model=tiiuae/falcon-7b-instruct,Kyle1668/boss-toxicity-t5-large,Kyle1668/boss-toxicity-bert --adaptive_model=NousResearch/Llama-2-7b-chat-hf --skip_style_model_eval --num_shots=16,0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

########## 10/23 Reruns ##########

boss_sentiment_ood_bert_icr:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=semval,dynasent --model=Kyle1668/boss-sentiment-bert-base-uncased --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

########## 10/20 Reruns ##########

ablate_model_size_paraphrase:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=test --model=EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,EleutherAI/pythia-12b --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=sst5 --model=EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,EleutherAI/pythia-12b --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=toxigen --model=EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,EleutherAI/pythia-12b --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb

irc_ood_news:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=test --model=Kyle1668/ag-news-bert-base-uncased,Kyle1668/ag-news-t5-large,tiiuae/falcon-7b-instruct --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

########## 10/14 Reruns ##########

# Falcon for semval and dynasent and T5 and BERT for the whole sweep
10_14_rerun_main_results_ood_sentiment:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=semval,dynasent --model=tiiuae/falcon-7b-instruct --adaptive_model=aug_insert,aug_substitute,aug_back-translate,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --model=Kyle1668/boss-sentiment-t5-large,Kyle1668/boss-sentiment-bert-base-uncased --adaptive_model=aug_insert,aug_substitute,aug_back-translate,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

# Falcon for implicit_hate and T5/BERT for the whole sweep
10_14_rerun_main_results_ood_toxicity:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=implicit_hate --model=tiiuae/falcon-7b-instruct --adaptive_model=aug_insert,aug_substitute,aug_back-translate,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --model=Kyle1668/boss-toxicity-t5-large,Kyle1668/boss-toxicity-bert-base-uncased --adaptive_model=aug_insert,aug_substitute,aug_back-translate,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

# Falcon ICR for test split and T5/BERT for the whole sweep
10_14_rerun_main_results_ood_news:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=test --model=tiiuae/falcon-7b-instruct --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --model=Kyle1668/ag-news-t5-large,Kyle1668/ag-news-bert-base-uncased --adaptive_model=aug_insert,aug_substitute,aug_back-translate,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0,16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

########## Targeted Experiments ##########

main_results_id_no_llm:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=validation --evaluate_id_adaptation --model=Kyle1668/boss-sentiment-bert-base-uncased,Kyle1668/boss-sentiment-t5-large --adaptive_model=aug_insert,aug_substitute --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=validation --evaluate_id_adaptation --model=Kyle1668/boss-toxicity-bert-base-uncased,Kyle1668/boss-toxicity-t5-large --adaptive_model=aug_insert,aug_substitute,aug_back-translate --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=validation --evaluate_id_adaptation --model=kyle1668/ag-news-bert-base-uncased,kyle1668/ag-news-t5-large --adaptive_model=aug_insert,aug_substitute,aug_back-translate --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

main_icr_results:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=test --model=tiiuae/falcon-7b-instruct,Kyle1668/ag-news-bert-base-uncased,Kyle1668/ag-news-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=toxigen,adv_civil,implicit_hate --model=tiiuae/falcon-7b-instruct,Kyle1668/boss-toxicity-bert-base-uncased,Kyle1668/boss-toxicity-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=sst5,semval,dynasent --model=tiiuae/falcon-7b-instruct,Kyle1668/boss-sentiment-bert-base-uncased,Kyle1668/boss-sentiment-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

main_icr_results_no_falcon:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=test --model=Kyle1668/ag-news-bert-base-uncased,Kyle1668/ag-news-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=toxigen,adv_civil,implicit_hate --model=Kyle1668/boss-toxicity-bert-base-uncased,Kyle1668/boss-toxicity-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=sst5,semval,dynasent --model=Kyle1668/boss-sentiment-bert-base-uncased,Kyle1668/boss-sentiment-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

main_paraphrase_results:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --model=tiiuae/falcon-7b-instruct,Kyle1668/boss-sentiment-bert-base-uncased,Kyle1668/boss-sentiment-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --model=tiiuae/falcon-7b-instruct,Kyle1668/boss-toxicity-bert-base-uncased,Kyle1668/boss-toxicity-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --model=tiiuae/falcon-7b-instruct,Kyle1668/ag-news-bert-base-uncased,Kyle1668/ag-news-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

falcon_main_results_ood_no_icr:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --model=tiiuae/falcon-7b-instruct --adaptive_model=aug_back-translate,aug_insert,aug_substitute,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --model=tiiuae/falcon-7b-instruct --adaptive_model=aug_back-translate,aug_insert,aug_substitute,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --model=tiiuae/falcon-7b-instruct --adaptive_model=aug_back-translate,aug_insert,aug_substitute,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

ablate_model_size:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=sst5 --model=EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,EleutherAI/pythia-12b --adaptive_model=aug_back-translate,aug_insert,aug_substitute,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=toxigen --model=EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,EleutherAI/pythia-12b --adaptive_model=aug_back-translate,aug_insert,aug_substitute,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=test --model=EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,EleutherAI/pythia-12b --adaptive_model=aug_back-translate,aug_insert,aug_substitute,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

rewriter_model_eval:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=sst5,semval,dynasent --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=toxigen,adv_civil,implicit_hate --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=test --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb


rewriter_model_eval_llama2:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --model=NousResearch/Llama-2-7b-hf --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --model=NousResearch/Llama-2-7b-hf --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --model=NousResearch/Llama-2-7b-hf --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=random,topk_nearest --temperature=0 --trim_exemplars --use_wandb

rewriter_model_eval_topk_nearest:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=sst5,semval,dynasent --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=toxigen,adv_civil,implicit_hate --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=topk_nearest --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=test --model=stabilityai/StableBeluga-7b --adaptive_model=aug_back-translate --skip_style_model_eval --skip_eval_styling --num_shots=16 --icl_method=topk_nearest --temperature=0 --trim_exemplars --use_wandb

imdb_rotten_tomatoes:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=imdb_rotten_tomatoes --model=Kyle1668/imdb-bert-base-uncased,Kyle1668/imdb-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16,0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

gpt3_baseline_eval:
	python evaluate_styling.py --dataset=ag_news_twitter --model=gpt-3.5-turbo --split=test --adaptive_model=aug_back-translate,aug_insert,aug_substitute --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	python evaluate_styling.py --dataset=boss_toxicity --split=toxigen --model=gpt-3.5-turbo --adaptive_model=aug_back-translate,aug_insert,aug_substitute --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	python evaluate_styling.py --dataset=boss_sentiment --split=sst5 --model=gpt-3.5-turbo --adaptive_model=aug_back-translate,aug_insert,aug_substitute --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb