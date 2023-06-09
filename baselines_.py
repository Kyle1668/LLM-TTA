def evaluate_test_time_augmentation(experiment_id, model_name, model, tokenizer, dataset_name, dataset, icl_method, num_shots=None):
    eval_set = "test+adaptive"