import json
import requests
from tqdm import tqdm

""" CHOOSE MODEL """

model_name = 'base'
url = 'http://localhost:9001/infer_sentence'

# model_name = 'xl'
# url = 'http://localhost:9002/infer_sentence'

# model_name = 'lora_sy'
# url = 'http://localhost:9003/infer_sentence'

# model_name = 'lora_hy'
# url = 'http://localhost:9004/infer_sentence'


""" CHOOSE DATASET """

# SYNTHETIC DATASET
dataset_name = 'synthetic_dataset'
with open('./data/raa_synthetic_dataset_aug_transformed_test.json', encoding='utf-8') as fin:
    test_data = json.load(fin)

# # HYBRID DATASET
# dataset_name = 'hybrid_dataset'
# with open('./data/raa_hybrid_dataset_aug_transformed_test.json', encoding='utf-8') as fin:
#     test_data = json.load(fin)

# Final evaluation name
evaluation_model_name = f'{model_name}_{dataset_name}'

def get_prediction(instance):
    if 'base' in model_name or 'xl' in model_name:
        if 'Is there a valid' in instance:
            instance = instance.replace(' ### Answer:', " Answer only using 'Yes' or 'No'. ### Answer:")
        elif 'introduced or created by the authors' in instance:
            instance = instance.replace(' ### Answer:', " Answer only using 'Yes' or 'No'. ### Answer:")
        elif 'used or adopted by the authors' in instance:
            instance = instance.replace(' ### Answer:', " Answer only using 'Yes' or 'No'. ### Answer:")
        elif 'List all the artifacts' in instance:
            instance = instance.replace(' ### Answer:', " Answer with a list of artifacts in the format 'artifact_type: artifact_name' separated by '|' tokens. ### Answer:")
        else:
            instance = instance.replace(' ### Answer:', " The answer must be a text span from the Snippet. If you can't answer the question then respond with 'N/A'. ### Answer:")
    json_data = {
        'text': instance,
        'gen_config': {
            'max_new_tokens': 256,
        }
    }
    return requests.post(url, json=json_data).json()['output'][0]


def calculate_exact_match(targets, predictions):
        errors = []

        total_instances = len(targets)
        match_count = 0

        for i, (target, prediction) in enumerate(zip(targets, predictions)):
            if target.lower() == prediction.lower():  # Case-insensitive comparison
                match_count += 1
            else:
                errors.append((i, target, prediction))

        exact_match_score = match_count / total_instances
        return exact_match_score, errors


def calculate_lenient_match(targets, predictions):
    errors = []

    total_instances = len(targets)
    match_count = 0

    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        if target.lower() in prediction.lower() or prediction.lower() in target.lower():  # Case-insensitive comparison
            match_count += 1
        else:
            errors.append((i, target, prediction))

    exact_match_score = match_count / total_instances
    return exact_match_score, errors


def calculate_prf_binary(targets, predictions):
    errors = []

    # Calculate the TP, FP, FN
    tp = 0
    fp = 0
    fn = 0

    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        if target == 1 and prediction == 1:
            tp += 1
        elif target == 1 and prediction == 0:
            fn += 1
            errors.append((i, target, prediction))
        elif target == 0 and prediction == 1:
            fp += 1
            errors.append((i, target, prediction))
    
    # Calculate the precision, recall, f1
    if tp + fp == 0:
        precision = 0  # Handle division by zero
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0  # Handle division by zero
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0  # Handle division by zero
    else:
        f1 = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1, errors


all_questions = set([x['input'].split('###')[2].strip()[10:] for x in test_data])


# Create a dictionary with the questions as keys and the instances as values
questions_to_instances = {}
for d_i, d in enumerate(test_data):
    question = d['input'].split('###')[2].strip()[10:]
    if question not in questions_to_instances:
        questions_to_instances[question] = []
    d['id'] = d_i
    questions_to_instances[question].append(d)

# Find the predictions + metrics for each question
questions_to_scores = {}
questions_to_scores_onlymetrics = {}
for question, instances in tqdm(questions_to_instances.items()):
    # Categorize the questions into:
    # 1. Yes / No Questions
    # 2. Extraction Questions

    # WE DO NOT EVALUATE THE CASES WHERE THE TARGET IS A LIST OF ARTIFACTS
    if sorted(set([x['output'] for x in instances if '|' not in x['output']])) == ['No', 'Yes']:
        q_category = 'yes_no'
    else:
        q_category = 'extraction'

    # print('Question:', question)
    ids = [x['id'] for x in instances]
    targets = []
    predictions = []
    for d in tqdm(instances):
        instance = d['input']
        target = d['output']
        # WE DO NOT EVALUATE THE CASES WHERE THE TARGET IS A LIST OF ARTIFACTS
        if '|' in target:
            continue
        prediction = get_prediction(instance)
        targets.append(target)
        predictions.append(prediction) 

    # Calculate the metrics between targets and predictions
    # NOTE: there are some cases where the one is in lowercase and the other in uppercase, so we lowercase everything

    if q_category == 'yes_no':
        questions_to_scores[question] = {
            'category': q_category,
            'ids': ids,
            'targets': targets,
            'predictions': predictions,
            'yes_no_binary': None,
        }
        questions_to_scores_onlymetrics[question] = {
            'category': q_category,
            'yes_no_binary': None
        }
        # Calculate PRF for Yes / No questions
        yes_no_targets = [x.lower()=='yes' for x in targets]
        yes_no_predictions = [x.lower()=='yes' for x in predictions]
        yes_no_precision, yes_no_recall, yes_no_f1, yes_no_errors = calculate_prf_binary(yes_no_targets, yes_no_predictions)
        # Correct the indices of the errors
        yes_no_errors = [(ids[x[0]], 'Yes' if x[1]==1 else 'No', 'Yes' if x[2] else 'No') for x in yes_no_errors]

        questions_to_scores[question]['yes_no_binary'] = {
            'precision': yes_no_precision,
            'recall': yes_no_recall,
            'f1': yes_no_f1,
            'errors': yes_no_errors
        }
        questions_to_scores_onlymetrics[question]['yes_no_binary'] = {
            'precision': yes_no_precision,
            'recall': yes_no_recall,
            'f1': yes_no_f1
        }
    else:
        questions_to_scores[question] = {
            'category': q_category,
            'ids': ids,
            'targets': targets,
            'predictions': predictions,
            'no_answer_binary': None,
            'w_answer': None,
            'all': None
        }
        questions_to_scores_onlymetrics[question] = {
            'category': q_category,
            'no_answer_binary': None,
            'w_answer': None,
            'all': None
        }
        # Calculate no_answer as a binary classification (1 if is N/A, 0 if not)
        no_answer_binary_instances = [(x[0]=='N/A', x[1]=='N/A') for x in zip(targets, predictions)]
        if no_answer_binary_instances:
            no_answer_targets, no_answer_predictions = zip(*no_answer_binary_instances)
            no_answer_precision, no_answer_recall, no_answer_f1, no_answer_errors = calculate_prf_binary(no_answer_targets, no_answer_predictions)
            # Correct the indices of the errors
            no_answer_errors = [(ids[x[0]], 'N/A' if x[1]==1 else targets[x[0]], 'N/A' if x[2] else predictions[x[0]]) for x in no_answer_errors]

            questions_to_scores[question]['no_answer_binary'] = {
                'precision': no_answer_precision,
                'recall': no_answer_recall,
                'f1': no_answer_f1,
                'errors': no_answer_errors
            }
            questions_to_scores_onlymetrics[question]['no_answer_binary'] = {
                'precision': no_answer_precision,
                'recall': no_answer_recall,
                'f1': no_answer_f1
            }

        # Calculate seperately the w_answer targets
        w_answer_instances = [(x[0], x[1]) for x in zip(targets, predictions) if x[0] != 'N/A']
        if w_answer_instances:
            w_answer_targets, w_answer_predictions = zip(*w_answer_instances)
            w_answer_em_score, w_answer_em_errors = calculate_exact_match(w_answer_targets, w_answer_predictions)
            w_answer_lm_score, w_answer_lm_errors = calculate_lenient_match(w_answer_targets, w_answer_predictions)
            # Correct the indices of the errors
            w_answer_em_errors = [(ids[x[0]], x[1], x[2]) for x in w_answer_em_errors]
            w_answer_lm_errors = [(ids[x[0]], x[1], x[2]) for x in w_answer_lm_errors]

            questions_to_scores[question]['w_answer'] = {
                'em': w_answer_em_score,
                'lm': w_answer_lm_score,
                'em_errors': w_answer_em_errors,
                'lm_errors': w_answer_lm_errors,
            }
            questions_to_scores_onlymetrics[question]['w_answer'] = {
                'em': w_answer_em_score,
                'lm': w_answer_lm_score,
            }
        
        # Calculate all together
        all_targets = targets
        all_predictions = predictions
        all_em_score, all_em_errors = calculate_exact_match(all_targets, all_predictions)
        all_lm_score, all_lm_errors = calculate_lenient_match(all_targets, all_predictions)
        # Correct the indices of the errors
        all_em_errors = [(ids[x[0]], x[1], x[2]) for x in all_em_errors]
        all_lm_errors = [(ids[x[0]], x[1], x[2]) for x in all_lm_errors]

        questions_to_scores[question]['all'] = {
            'em': all_em_score,
            'lm': all_lm_score,
            'em_errors': all_em_errors,
            'lm_errors': all_lm_errors,
        }
        questions_to_scores_onlymetrics[question]['all'] = {
            'em': all_em_score,
            'lm': all_lm_score,
        }

""" Calculate the overall metrics """

# NOTE: we should cetagorize based on:
# - validity
# - metadata (name, version, license, URL)
# - ownership
# - usage

overall_metrics = {
    'validity': dict(),
    'metadata': {
        'name': dict(),
        'version': dict(),
        'license': dict(),
        'url': dict(),
        'all': dict()
    },
    'ownership': dict(),
    'usage': dict()
}

overall_metrics_onlymetrics = {
    'validity': dict(),
    'metadata': {
        'name': dict(),
        'version': dict(),
        'license': dict(),
        'url': dict(),
        'all': dict()
    },
    'ownership': dict(),
    'usage': dict()
}

metric_to_question_query = {
    'validity': 'Is there a valid',
    'metadata': {
        'name': 'What is the name',
        'version': 'What is the version',
        'license': 'What is the license',
        'url': 'What is the URL'
    },
    'ownership': 'introduced or created by the authors',
    'usage': 'used or adopted by the authors'
}

metric_to_type = {
    'validity': 'yes_no',
    'metadata': {
        'name': 'extraction',
        'version': 'extraction',
        'license': 'extraction',
        'url': 'extraction'
    },
    'ownership': 'yes_no',
    'usage': 'yes_no'
}

# Calculate each metric
for metric, question_query in metric_to_question_query.items():
    if isinstance(question_query, dict):
        for sub_metric, sub_question_query in question_query.items():
            # Get the questions and scores that match the query
            questions, scores = zip(*[q for q in questions_to_scores.items() if sub_question_query in q[0]])
            # Get the targets and predictions for these questions
            ids = [s['ids'] for s in scores]
            targets = [s['targets'] for s in scores]
            predictions = [s['predictions'] for s in scores]

            # Flatten the lists
            ids = [item for sublist in ids for item in sublist]
            targets = [item for sublist in targets for item in sublist]
            predictions = [item for sublist in predictions for item in sublist]

            # Calculate the metric
            if metric_to_type[metric][sub_metric] == 'yes_no':
                # Calculate PRF for Yes / No questions
                yes_no_targets = [x.lower()=='yes' for x in targets]
                yes_no_predictions = [x.lower()=='yes' for x in predictions]
                yes_no_precision, yes_no_recall, yes_no_f1, yes_no_errors = calculate_prf_binary(yes_no_targets, yes_no_predictions)
                # Correct the indices of the errors
                yes_no_errors = [(ids[x[0]], 'Yes' if x[1]==1 else 'No', 'Yes' if x[2]==1 else 'No') for x in yes_no_errors]

                overall_metrics[metric][sub_metric]['no_answer_binary'] = {
                    'precision': yes_no_precision,
                    'recall': yes_no_recall,
                    'f1': yes_no_f1,
                    'errors': yes_no_errors
                }
                overall_metrics_onlymetrics[metric][sub_metric]['no_answer_binary'] = {
                    'precision': yes_no_precision,
                    'recall': yes_no_recall,
                    'f1': yes_no_f1,
                }
            else:
                # Calculate no_answer as a binary classification (1 if is N/A, 0 if not)
                no_answer_binary_instances = [(x[0]=='N/A', x[1]=='N/A') for x in zip(targets, predictions)]
                if no_answer_binary_instances:
                    no_answer_targets, no_answer_predictions = zip(*no_answer_binary_instances)
                    no_answer_precision, no_answer_recall, no_answer_f1, no_answer_errors = calculate_prf_binary(no_answer_targets, no_answer_predictions)
                    # Correct the indices of the errors
                    no_answer_errors = [(ids[x[0]], 'N/A' if x[1]==1 else targets[x[0]], 'N/A' if x[2] else predictions[x[0]]) for x in no_answer_errors]

                    overall_metrics[metric][sub_metric]['no_answer_binary'] = {
                        'precision': no_answer_precision,
                        'recall': no_answer_recall,
                        'f1': no_answer_f1,
                        'errors': no_answer_errors
                    }
                    overall_metrics_onlymetrics[metric][sub_metric]['no_answer_binary'] = {
                        'precision': no_answer_precision,
                        'recall': no_answer_recall,
                        'f1': no_answer_f1,
                    }

                # Calculate seperately the w_answer targets
                w_answer_instances = [(x[0], x[1]) for x in zip(targets, predictions) if x[0] != 'N/A']
                if w_answer_instances:
                    w_answer_targets, w_answer_predictions = zip(*w_answer_instances)
                    w_answer_em_score, w_answer_em_errors = calculate_exact_match(w_answer_targets, w_answer_predictions)
                    w_answer_lm_score, w_answer_lm_errors = calculate_lenient_match(w_answer_targets, w_answer_predictions)
                    # Correct the indices of the errors
                    w_answer_em_errors = [(ids[x[0]], x[1], x[2]) for x in w_answer_em_errors]
                    w_answer_lm_errors = [(ids[x[0]], x[1], x[2]) for x in w_answer_lm_errors]

                    overall_metrics[metric][sub_metric]['w_answer'] = {
                        'em': w_answer_em_score,
                        'lm': w_answer_lm_score,
                        'em_errors': w_answer_em_errors,
                        'lm_errors': w_answer_lm_errors,
                    }
                    overall_metrics_onlymetrics[metric][sub_metric]['w_answer'] = {
                        'em': w_answer_em_score,
                        'lm': w_answer_lm_score,
                    }
                
                # Calculate all together
                all_targets = targets
                all_predictions = predictions
                all_em_score, all_em_errors = calculate_exact_match(all_targets, all_predictions)
                all_lm_score, all_lm_errors = calculate_lenient_match(all_targets, all_predictions)

                overall_metrics[metric][sub_metric]['all'] = {
                    'em': all_em_score,
                    'lm': all_lm_score,
                    'em_errors': all_em_errors,
                    'lm_errors': all_lm_errors,
                }
                overall_metrics_onlymetrics[metric][sub_metric]['all'] = {
                    'em': all_em_score,
                    'lm': all_lm_score,
                }
    else:
        # Get the questions and scores that match the query
        questions, scores = zip(*[q for q in questions_to_scores.items() if question_query in q[0]])
        # Get the targets and predictions for these questions
        ids = [s['ids'] for s in scores]
        targets = [s['targets'] for s in scores]
        predictions = [s['predictions'] for s in scores]

        # Flatten the lists
        ids = [item for sublist in ids for item in sublist]
        targets = [item for sublist in targets for item in sublist]
        predictions = [item for sublist in predictions for item in sublist]

        # Calculate the metric
        if metric_to_type[metric] == 'yes_no':
            # Calculate PRF for Yes / No questions
            yes_no_targets = [x.lower()=='yes' for x in targets]
            yes_no_predictions = [x.lower()=='yes' for x in predictions]
            yes_no_precision, yes_no_recall, yes_no_f1, yes_no_errors = calculate_prf_binary(yes_no_targets, yes_no_predictions)
            # Correct the indices of the errors
            yes_no_errors = [(ids[x[0]], 'Yes' if x[1]==1 else 'No', 'Yes' if x[2] else 'No') for x in yes_no_errors]

            overall_metrics[metric]['yes_no_binary'] = {
                'precision': yes_no_precision,
                'recall': yes_no_recall,
                'f1': yes_no_f1,
                'errors': yes_no_errors
            }
            overall_metrics_onlymetrics[metric]['yes_no_binary'] = {
                'precision': yes_no_precision,
                'recall': yes_no_recall,
                'f1': yes_no_f1,
            }
        elif metric_to_type[metric] == 'extraction':
            # Calculate no_answer as a binary classification (1 if is N/A, 0 if not)
            no_answer_binary_instances = [(x[0]=='N/A', x[1]=='N/A') for x in zip(targets, predictions)]
            if no_answer_binary_instances:
                no_answer_targets, no_answer_predictions = zip(*no_answer_binary_instances)
                no_answer_precision, no_answer_recall, no_answer_f1, no_answer_errors = calculate_prf_binary(no_answer_targets, no_answer_predictions)
                # Correct the indices of the errors
                no_answer_errors = [(ids[x[0]], 'N/A' if x[1]==1 else targets[x[0]], 'N/A' if x[2] else predictions[x[0]]) for x in no_answer_errors]

                overall_metrics[metric]['no_answer_binary'] = {
                    'precision': no_answer_precision,
                    'recall': no_answer_recall,
                    'f1': no_answer_f1,
                    'errors': no_answer_errors
                }
                overall_metrics_onlymetrics[metric]['no_answer_binary'] = {
                    'precision': no_answer_precision,
                    'recall': no_answer_recall,
                    'f1': no_answer_f1,
                }

            # Calculate seperately the w_answer targets
            w_answer_instances = [(x[0], x[1]) for x in zip(targets, predictions) if x[0] != 'N/A']
            if w_answer_instances:
                w_answer_targets, w_answer_predictions = zip(*w_answer_instances)
                w_answer_em_score, w_answer_em_errors = calculate_exact_match(w_answer_targets, w_answer_predictions)
                w_answer_lm_score, w_answer_lm_errors = calculate_lenient_match(w_answer_targets, w_answer_predictions)
                # Correct the indices of the errors
                w_answer_em_errors = [(ids[x[0]], x[1], x[2]) for x in w_answer_em_errors]
                w_answer_lm_errors = [(ids[x[0]], x[1], x[2]) for x in w_answer_lm_errors]

                overall_metrics[metric]['w_answer'] = {
                    'em': w_answer_em_score,
                    'lm': w_answer_lm_score,
                    'em_errors': w_answer_em_errors,
                    'lm_errors': w_answer_lm_errors,
                }
                overall_metrics_onlymetrics[metric]['w_answer'] = {
                    'em': w_answer_em_score,
                    'lm': w_answer_lm_score,
                }
            
            # Calculate all together
            all_targets = targets
            all_predictions = predictions
            all_em_score, all_em_errors = calculate_exact_match(all_targets, all_predictions)
            all_lm_score, all_lm_errors = calculate_lenient_match(all_targets, all_predictions)

            overall_metrics[metric]['all'] = {
                'em': all_em_score,
                'lm': all_lm_score,
                'em_errors': all_em_errors,
                'lm_errors': all_lm_errors,
            }
            overall_metrics_onlymetrics[metric]['all'] = {
                'em': all_em_score,
                'lm': all_lm_score,
            }

# Save the results
with open(f'./evaluation_results/test_results_{evaluation_model_name}.json', 'w', encoding='utf-8') as f:
    json.dump({
        'questions_to_scores': questions_to_scores,
        'overall_metrics': overall_metrics,
    }, f, indent=1)

with open(f'./evaluation_results/test_results_{evaluation_model_name}_onlymetrics.json', 'w', encoding='utf-8') as f:
    json.dump({
        'questions_to_scores': questions_to_scores_onlymetrics,
        'overall_metrics': overall_metrics_onlymetrics,
    }, f, indent=1)
