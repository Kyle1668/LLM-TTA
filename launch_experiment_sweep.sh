main_results_experiment_ids=(
    # Main Results Split by ID & OOD
    "main_results_ood_sentiment"
    "main_results_ood_toxicity"
    "main_results_ood_news"
    "main_results_id_sentiment"
    "main_results_id_toxicity"
    "main_results_id_news"

    # Main Results
    # "main_results_sentiment"
    # "main_results_toxicity"
    # "main_results_news"

    # Training set size ablations
    
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
num_gpus=6
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

# grid search over the hyperparameters
for experiment_id in "${main_results_experiment_ids[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "sbatch $gpu_arg run_experiment.sh $experiment_id $seed"
        sbatch $gpu_arg run_experiment.sh $experiment_id $seed
    done
done

