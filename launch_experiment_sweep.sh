main_results_experiment_ids=(
    # Main Results Split by ID & OOD
    # "main_results_ood_sentiment"
    # "main_results_ood_toxicity"
    # "main_results_ood_news"
    main_results_ood_sentiment_translate
    main_results_ood_toxicity_translate
    main_results_ood_news_translate

    "main_results_id_news"
    "main_results_id_sentiment"
    "main_results_id_toxicity"

    # Main Results
    # "main_results_sentiment"
    # "main_results_toxicity"
    # "main_results_news"

    # Training set size ablations
    # "ablate_data_sentiment"
    # "ablate_data_toxicity"
    # "ablate_data_ag_news"
)

# Generated between 1-100 using Google
seeds=(
    3
    17
    46
    58
    90
)

gpu=$1
num_gpus=$2
gpu_arg=""
# if gpu = a100
if [ $gpu = "a100" ]; then
    gpu_arg="--gres=gpu:a100:$num_gpus --constraint=a100_80gb"
elif [ $gpu = "a6000" ]; then
    gpu_arg="--gres=gpu:a6000:$num_gpus"
else
    echo "Invalid GPU"
    exit 1
fi

is_valid_num_gpus=$(echo $num_gpus | grep -E "^[0-9]+$")
if [ -z $is_valid_num_gpus ]; then
    echo "Invalid number of GPUs"
    exit 1
fi

# grid search over the hyperparameters
for experiment_id in "${main_results_experiment_ids[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "sbatch $gpu_arg run_experiment.sh $experiment_id $seed"
        sbatch $gpu_arg run_experiment.sh $experiment_id $seed
    done
done

