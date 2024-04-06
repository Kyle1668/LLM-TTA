main_results_experiment_ids=(
    # Main Results
    "main_results_sentiment"
    "main_results_toxicity"
    "main_results_news"
)

# Generated between 1-100 using Google
seeds=(
    3
    17
    46
    58
    90
)

# grid search over the hyperparameters
for experiment_id in "${main_results_experiment_ids[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "sbatch run_experiment.sh $experiment_id $seed"
        sbatch run_experiment.sh $experiment_id $seed
    done
done

