import json

create_predictions_files = False

if create_predictions_files:
    # Synthetic Dataset

    with open('./data/raa_synthetic_dataset_aug_transformed_test.json') as f:
        synthetic_data = json.load(f)

    with open('./evaluation_results/test_results_base_synthetic_dataset.json') as f:
        base_synthetic = json.load(f)

    with open('./evaluation_results/test_results_xl_synthetic_dataset.json') as f:
        xl_synthetic = json.load(f)

    with open('./evaluation_results/test_results_lora_sy_synthetic_dataset.json') as f:
        synthetic_synthetic = json.load(f)

    with open('./evaluation_results/test_results_lora_hy_synthetic_dataset.json') as f:
        hybrid_synthetic = json.load(f)

    questions_to_instances = {}
    for d_i, d in enumerate(synthetic_data):
        question = d['input'].split('###')[2].strip()[10:]
        if question not in questions_to_instances:
            questions_to_instances[question] = []
        d['id'] = d_i
        questions_to_instances[question].append(d)

    valid_ids = []
    for question, instances in questions_to_instances.items():
        for d in instances:
            instance = d['input']
            target = d['output']
            # WE DO NOT EVALUATE THE CASES WHERE THE TARGET IS A LIST OF ARTIFACTS
            if '|' in target:
                continue
            else:
                valid_ids.append(d['id'])
    valid_ids = set(valid_ids)

    # Get the predictions for each model
    base_synthetic_predictions = sorted([i for s in [list(zip([x for x in base_synthetic['questions_to_scores'][q]['ids'] if x in valid_ids], base_synthetic['questions_to_scores'][q]['predictions'])) for q in base_synthetic['questions_to_scores']] for i in s], key=lambda z:z[0])
    xl_synthetic_predictions = sorted([i for s in [list(zip([x for x in xl_synthetic['questions_to_scores'][q]['ids'] if x in valid_ids], xl_synthetic['questions_to_scores'][q]['predictions'])) for q in xl_synthetic['questions_to_scores']] for i in s], key=lambda z:z[0])
    synthetic_synthetic_predictions = sorted([i for s in [list(zip([x for x in synthetic_synthetic['questions_to_scores'][q]['ids'] if x in valid_ids], synthetic_synthetic['questions_to_scores'][q]['predictions'])) for q in synthetic_synthetic['questions_to_scores']] for i in s], key=lambda z:z[0])
    hybrid_synthetic_predictions = sorted([i for s in [list(zip([x for x in hybrid_synthetic['questions_to_scores'][q]['ids'] if x in valid_ids], hybrid_synthetic['questions_to_scores'][q]['predictions'])) for q in hybrid_synthetic['questions_to_scores']] for i in s], key=lambda z:z[0])

    # Create a dictionary of id to predictions for each model
    base_synthetic_predictions_dict = {x[0]: x[1] for x in base_synthetic_predictions}
    xl_synthetic_predictions_dict = {x[0]: x[1] for x in xl_synthetic_predictions}
    synthetic_synthetic_predictions_dict = {x[0]: x[1] for x in synthetic_synthetic_predictions}
    hybrid_synthetic_predictions_dict = {x[0]: x[1] for x in hybrid_synthetic_predictions}

    # Create a list of all instances in the dataset with the predictions of each model
    synthetic_data_with_predictions = []
    for i in range(len(synthetic_data)):
        instance = {
            'id': i,
            'snippet': synthetic_data[i]['input'].split('### ')[1].split('Snippet:')[1].strip(),
            'question': synthetic_data[i]['input'].split('### ')[2].split('Question:')[1].strip(),
            'answer': synthetic_data[i]['output']
        }
        instance['base_prediction'] = base_synthetic_predictions_dict[i] if i in base_synthetic_predictions_dict else None
        instance['xl_prediction'] = xl_synthetic_predictions_dict[i] if i in xl_synthetic_predictions_dict else None
        instance['synthetic_prediction'] = synthetic_synthetic_predictions_dict[i] if i in synthetic_synthetic_predictions_dict else None
        instance['hybrid_prediction'] = hybrid_synthetic_predictions_dict[i] if i in hybrid_synthetic_predictions_dict else None
        synthetic_data_with_predictions.append(instance)

    # Hybrid Dataset

    with open('./data/raa_hybrid_dataset_aug_transformed_test.json') as f:
        hybrid_data = json.load(f)

    with open('./evaluation_results/test_results_base_hybrid_dataset.json') as f:
        base_hybrid = json.load(f)

    with open('./evaluation_results/test_results_xl_hybrid_dataset.json') as f:
        xl_hybrid = json.load(f)

    with open('./evaluation_results/test_results_lora_sy_hybrid_dataset.json') as f:
        synthetic_hybrid = json.load(f)

    with open('./evaluation_results/test_results_lora_hy_hybrid_dataset.json') as f:
        hybrid_hybrid = json.load(f)

    questions_to_instances = {}
    for d_i, d in enumerate(hybrid_data):
        question = d['input'].split('###')[2].strip()[10:]
        if question not in questions_to_instances:
            questions_to_instances[question] = []
        d['id'] = d_i
        questions_to_instances[question].append(d)

    valid_ids = []
    for question, instances in questions_to_instances.items():
        for d in instances:
            instance = d['input']
            target = d['output']
            # WE DO NOT EVALUATE THE CASES WHERE THE TARGET IS A LIST OF ARTIFACTS
            if '|' in target:
                continue
            else:
                valid_ids.append(d['id'])
    valid_ids = set(valid_ids)

    # Get the predictions for each model
    base_hybrid_predictions = sorted([i for s in [list(zip([x for x in base_hybrid['questions_to_scores'][q]['ids'] if x in valid_ids], base_hybrid['questions_to_scores'][q]['predictions'])) for q in base_hybrid['questions_to_scores']] for i in s], key=lambda z:z[0])
    xl_hybrid_predictions = sorted([i for s in [list(zip([x for x in xl_hybrid['questions_to_scores'][q]['ids'] if x in valid_ids], xl_hybrid['questions_to_scores'][q]['predictions'])) for q in xl_hybrid['questions_to_scores']] for i in s], key=lambda z:z[0])
    synthetic_hybrid_predictions = sorted([i for s in [list(zip([x for x in synthetic_hybrid['questions_to_scores'][q]['ids'] if x in valid_ids], synthetic_hybrid['questions_to_scores'][q]['predictions'])) for q in synthetic_hybrid['questions_to_scores']] for i in s], key=lambda z:z[0])
    hybrid_hybrid_predictions = sorted([i for s in [list(zip([x for x in hybrid_hybrid['questions_to_scores'][q]['ids'] if x in valid_ids], hybrid_hybrid['questions_to_scores'][q]['predictions'])) for q in hybrid_hybrid['questions_to_scores']] for i in s], key=lambda z:z[0])

    # Create a dictionary of id to predictions for each model
    base_hybrid_predictions_dict = {x[0]: x[1] for x in base_hybrid_predictions}
    xl_hybrid_predictions_dict = {x[0]: x[1] for x in xl_hybrid_predictions}
    synthetic_hybrid_predictions_dict = {x[0]: x[1] for x in synthetic_hybrid_predictions}
    hybrid_hybrid_predictions_dict = {x[0]: x[1] for x in hybrid_hybrid_predictions}

    # Create a list of all instances in the dataset with the predictions of each model
    hybrid_data_with_predictions = []
    for i in range(len(hybrid_data)):
        instance = {
            'id': i,
            'snippet': hybrid_data[i]['input'].split('### ')[1].split('Snippet:')[1].strip(),
            'question': hybrid_data[i]['input'].split('### ')[2].split('Question:')[1].strip(),
            'answer': hybrid_data[i]['output']
        }
        instance['base_prediction'] = base_hybrid_predictions_dict[i] if i in base_hybrid_predictions_dict else None
        instance['xl_prediction'] = xl_hybrid_predictions_dict[i] if i in xl_hybrid_predictions_dict else None
        instance['synthetic_prediction'] = synthetic_hybrid_predictions_dict[i] if i in synthetic_hybrid_predictions_dict else None
        instance['hybrid_prediction'] = hybrid_hybrid_predictions_dict[i] if i in hybrid_hybrid_predictions_dict else None
        hybrid_data_with_predictions.append(instance)

    # Save the results
    with open('./evaluation_results/synthetic_data_with_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(synthetic_data_with_predictions, f, indent=1)

    with open('./evaluation_results/hybrid_data_with_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(hybrid_data_with_predictions, f, indent=1)

    exit()

# Inspect

with open('./evaluation_results/synthetic_data_with_predictions.json', encoding='utf-8') as f:
    synthetic_data_with_predictions = json.load(f)

with open('./evaluation_results/hybrid_data_with_predictions.json', encoding='utf-8') as f:
    hybrid_data_with_predictions = json.load(f)

def check_equal_em(target, prediction):
    if prediction is None:
        return False
    return target.lower() == prediction.lower()

def check_equal_lm(target, prediction):
    if prediction is None:
        return False
    return target.lower() in prediction.lower() or prediction.lower() in target.lower()

h_synthetic_model_diff_per_q_em = {k: [x for x in hybrid_data_with_predictions if x['question'] == k and check_equal_em(x['answer'], x['hybrid_prediction']) and not check_equal_em(x['answer'], x['synthetic_prediction'])] for k in sorted(set([y['question'] for y in hybrid_data_with_predictions]))}
h_hybrid_model_diff_per_q_em = {k: [x for x in hybrid_data_with_predictions if x['question'] == k and check_equal_em(x['answer'], x['synthetic_prediction']) and not check_equal_em(x['answer'], x['hybrid_prediction'])] for k in sorted(set([y['question'] for y in hybrid_data_with_predictions]))}

print()
