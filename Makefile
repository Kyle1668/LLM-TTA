install_depends:
	conda create -n eval-aug python=3.10
	conda activate eval-aug
	pip install -r requirements.txt

download_boss_benchmark:
	mkdir datasets; cd datasets && wget https://huggingface.co/datasets/Kyle1668/BOSS-Robustness-Benchmark/resolve/main/BOSS.zip && unzip BOSS.zip
	mv datasets/process datasets/boss_benchmark

main_icr_results:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --model=Kyle1668/boss-sentiment-bert-base-uncased,Kyle1668/boss-sentiment-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16,0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --model=Kyle1668/boss-toxicity-bert-base-uncased,Kyle1668/boss-toxicity-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16,0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --model=Kyle1668/ag-news-bert-base-uncased,Kyle1668/ag-news-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16,0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

main_paraphrase_results:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --model=Kyle1668/boss-sentiment-bert-base-uncased,Kyle1668/boss-sentiment-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --model=Kyle1668/boss-toxicity-bert-base-uncased,Kyle1668/boss-toxicity-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --model=Kyle1668/ag-news-bert-base-uncased,Kyle1668/ag-news-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

ablate_model_size:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --split=sst5 --model=EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,EleutherAI/pythia-12b --adaptive_model=aug_back-translate,aug_insert,aug_substitute,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --split=toxigen --model=EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,EleutherAI/pythia-12b --adaptive_model=aug_back-translate,aug_insert,aug_substitute,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --split=test --model=EleutherAI/pythia-1.4b,EleutherAI/pythia-2.8b,EleutherAI/pythia-6.9b,EleutherAI/pythia-12b --adaptive_model=aug_back-translate,aug_insert,aug_substitute,stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

boss_sentiment:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_sentiment --model=Kyle1668/boss-sentiment-bert-base-uncased,Kyle1668/boss-sentiment-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16,0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

boss_toxicity:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=boss_toxicity --model=Kyle1668/boss-toxicity-bert-base-uncased,Kyle1668/boss-toxicity-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16,0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

ag_news_twitter:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=ag_news_twitter --model=Kyle1668/ag-news-bert-base-uncased,Kyle1668/ag-news-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16,0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

imdb_rotten_tomatoes:
	torchrun --nproc-per-node=gpu evaluate_styling.py --dataset=imdb_rotten_tomatoes --model=Kyle1668/imdb-bert-base-uncased,Kyle1668/imdb-t5-large --adaptive_model=stabilityai/StableBeluga-7b --skip_style_model_eval --num_shots=16,0 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

gpt3_baseline_eval:
	python evaluate_styling.py --dataset=ag_news_twitter --model=gpt-3.5-turbo --split=test --adaptive_model=aug_back-translate,aug_insert,aug_substitute --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	python evaluate_styling.py --dataset=boss_toxicity --split=toxigen --model=gpt-3.5-turbo --adaptive_model=aug_back-translate,aug_insert,aug_substitute --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	python evaluate_styling.py --dataset=boss_sentiment --split=sst5 --model=gpt-3.5-turbo --adaptive_model=aug_back-translate,aug_insert,aug_substitute --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb

pythia_id_evals:
	python evaluate_styling.py --dataset=imdb_rotten_tomatoes --split=validation --model=EleutherAI/pythia-6.9b --adaptive_model=aug_back-translate,aug_insert,aug_substitute --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	python evaluate_styling.py --dataset=ag_news_twitter --split=validation --model=EleutherAI/pythia-6.9b --adaptive_model=aug_back-translate,aug_insert,aug_substitute --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	python evaluate_styling.py --dataset=boss_toxicity --split=validation --model=EleutherAI/pythia-6.9b --adaptive_model=aug_back-translate,aug_insert,aug_substitute --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb
	python evaluate_styling.py --dataset=boss_sentiment --split=validation --model=EleutherAI/pythia-6.9b --adaptive_model=aug_back-translate,aug_insert,aug_substitute --skip_style_model_eval --num_shots=16 --icl_method=random --temperature=0 --trim_exemplars --use_wandb