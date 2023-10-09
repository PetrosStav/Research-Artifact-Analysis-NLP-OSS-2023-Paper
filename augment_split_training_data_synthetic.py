import re
import json
import random
import itertools
import numpy as np
from copy import deepcopy
from tqdm import tqdm, trange
from collections import Counter, OrderedDict

random.seed(1993)

def create_ordered_set(items):
    return list(OrderedDict.fromkeys(items))

# Load the latest data
with open('./data/raa_synthetic_dataset.json', encoding='utf-8') as fin:
    all_data = json.load(fin)

""" MAP TYPES """
type_mapping = {
    'method': 'software',
    'repository': 'dataset',
    'dataset': 'dataset',
    'software': 'software',
}

for d in all_data:
    d['Type'] = type_mapping[d['Type']]

""" REPLACING ARTIFACT NAMES (LOWERCASE) """
# NOTE: To avoid errors, this must be done on the unique snippets, NOT on the instances

# See the names + frequencies
name_frequencies = Counter([x['Name'] for x in all_data if 'Name' in x]).most_common()

# Perform sampling to replace a 15% of the artifact names with their lowercase version
# NOTE: this percentage will affect a higher percentage of the instances, since some snippets have more than one artifact and we will replace all of them
#       for that reason we keep it low (we have calculated that it is approx 31.6% change in the instances)
snippets_to_change = []
for d in all_data:
    if 'Name' in d and 'N/A' not in d['Name'] and random.random() < 0.15:
        snippets_to_change.append([d['Snippet'].replace('<m>', '').replace('</m>', ''), d['Name']])
        if '|' in d['Name']:
            for name in d['Name'].split('|'):
                d['Snippet'] = d['Snippet'].replace(name.strip(), name.strip().lower())
        else:
            d['Snippet'] = d['Snippet'].replace(d['Name'], d['Name'].lower())
        d['Name'] = d['Name'].lower()

# Go though the snippets to change and replace the artifact names with their lowercase version
for d in all_data:
    for snippet, name in snippets_to_change:
        if d['Snippet'].replace('<m>', '').replace('</m>', '') == snippet:
            if '|' in name:
                for n in name.split('|'):
                    d['Snippet'] = d['Snippet'].replace(n.strip(), n.strip().lower())
            else:
                d['Snippet'] = d['Snippet'].replace(name, name.lower())
            if 'Name' in d and 'N/A' not in d['Name'] and d['Name'].lower() == name.lower():
                d['Name'] = name.lower()

percentage_changed = len([x for x in all_data if x['Valid']=='Yes' and x['Name']==x['Name'].lower()]) / (len([x for x in all_data if x['Valid']=='Yes' and x['Name']!=x['Name'].lower()]) + len([x for x in all_data if x['Valid']=='Yes' and x['Name']==x['Name'].lower()]))
print('Percentage of changed artifact names: {}'.format(percentage_changed))

# Get unique snippets
all_unique_snippets = sorted(set([x['Snippet'].replace('<m>', '').replace('</m>', '') for x in all_data]))

# Dictionary that maps the unique snippets to the data
unique_snippets_to_data = {}
for d in all_data:
    snippet = d['Snippet'].replace('<m>', '').replace('</m>', '')
    if snippet not in unique_snippets_to_data:
        unique_snippets_to_data[snippet] = [d]
    else:
        unique_snippets_to_data[snippet].append(d)

# Create data dictionary (we do this after the replacement to avoid errors)
all_data_dict = {}
for d in all_data:
    all_data_dict[d['Snippet']] = d

""" SPLIT THE DATA INTO TRAINING, EVAL AND TEST SETS """

# We need to create balanced sets in terms of:
#  - artifact count
#  - artifact types
#  - artifact metadata (name + lowercase, version, license, url, ownership, usage)
#  - special-type instances (this is automatically balanced by the artifact count -- CHECK)
# NOTE: we need to do that keeping the instances of the same snippet together


def calculate_imbalance(train_counter, dev_counter, test_counter):
    # A small value to avoid division by zero
    epsilon = 1e-8

    """ INTRA """
    # Get the set of all elements across all splits
    all_element_counts = set(train_counter.keys()) | set(dev_counter.keys()) | set(test_counter.keys())
    
    # Calculate the intra-class proportions for each split, handling missing counts
    train_proportions = np.array([train_counter.get(t, 0) for t in all_element_counts]) / sum(train_counter.values())
    dev_proportions = np.array([dev_counter.get(t, 0) for t in all_element_counts]) / sum(dev_counter.values())
    test_proportions = np.array([test_counter.get(t, 0) for t in all_element_counts]) / sum(test_counter.values())

    # Calculate the intra-class imbalance ratios
    train_ratio = (np.max(train_proportions) + epsilon) / (np.min(train_proportions) + epsilon)
    dev_ratio = (np.max(dev_proportions) + epsilon) / (np.min(dev_proportions) + epsilon)
    test_ratio = (np.max(test_proportions) + epsilon) / (np.min(test_proportions) + epsilon)

    # Calculate the intra-class imbalance score as the average of the imbalance ratios
    intra_balance_score = np.mean([train_ratio, dev_ratio, test_ratio])

    """ INTER """
    # Create dictionaries from element to train, dev, and test counts
    element_counts_counts = {}
    for t in all_element_counts:
        element_counts_counts[t] = [
            train_counter.get(t, 0),
            dev_counter.get(t, 0),
            test_counter.get(t, 0)
        ]
    
    # Calculate the inter-class proportions
    inter_class_proportions = []
    for element_count in element_counts_counts:
        inter_class_proportions.append(np.array(element_counts_counts[element_count]) / sum(element_counts_counts[element_count]))
    
    # Calculate the inter-class imbalance ratios
    inter_class_imbalance_ratios = []
    for proportions in inter_class_proportions:
        inter_class_imbalance_ratios.append((np.max(proportions) + epsilon) / (np.min(proportions) + epsilon))
    
    # Calculate the balance score as the average of the inter-class imbalance ratios
    inter_balance_score = np.mean(inter_class_imbalance_ratios)

    """ OVERALL """
    # Calculate the balance score as the average of the intra-class and inter-class imbalance ratios
    balance_score = np.mean([intra_balance_score, inter_balance_score])

    return balance_score


def score_split(train_snippets, dev_snippets, test_snippets):
    # Find the data for each split
    train_data = []
    for snippet in train_snippets:
        train_data.extend(unique_snippets_to_data[snippet])
    dev_data = []
    for snippet in dev_snippets:
        dev_data.extend(unique_snippets_to_data[snippet])
    test_data = []
    for snippet in test_snippets:
        test_data.extend(unique_snippets_to_data[snippet])

    """ TYPES BALANCE """
    # Check if the split is balanced in terms of artifact types
    train_artifact_types = Counter([x['Type'] for x in train_data])
    dev_artifact_types = Counter([x['Type'] for x in dev_data])
    test_artifact_types = Counter([x['Type'] for x in test_data])

    artifact_types_balance_score = calculate_imbalance(train_artifact_types, dev_artifact_types, test_artifact_types)

    """ ARTIFACT COUNT BALANCE """
    # Check if the split is balanced in terms of valid artifact count
    train_instances_per_artifact_count = Counter([len([x2 for x2 in unique_snippets_to_data[x] if x2['Valid']=='Yes']) for x in train_snippets])
    dev_instances_per_artifact_count = Counter([len([x2 for x2 in unique_snippets_to_data[x] if x2['Valid']=='Yes']) for x in dev_snippets])
    test_instances_per_artifact_count = Counter([len([x2 for x2 in unique_snippets_to_data[x] if x2['Valid']=='Yes']) for x in test_snippets])

    artifact_count_balance_score = calculate_imbalance(train_instances_per_artifact_count, dev_instances_per_artifact_count, test_instances_per_artifact_count)

    """ ARTIFACT METADATA BALANCE """
    # Check if the split is balanced in terms of artifact metadata (name + lowercase, version, license, url, ownership, usage)
    train_artifact_name = Counter(['N/A' not in x['Name'] for x in train_data if x['Valid']=='Yes'])
    dev_artifact_name = Counter(['N/A' not in x['Name'] for x in dev_data if x['Valid']=='Yes'])
    test_artifact_name = Counter(['N/A' not in x['Name'] for x in test_data if x['Valid']=='Yes'])

    artifact_name_balance_score = calculate_imbalance(train_artifact_name, dev_artifact_name, test_artifact_name)

    train_artifact_name_lowercase = Counter([x['Name']==x['Name'].lower() for x in train_data if x['Valid']=='Yes' and 'N/A' not in x['Name']])
    dev_artifact_name_lowercase = Counter([x['Name']==x['Name'].lower() for x in dev_data if x['Valid']=='Yes' and 'N/A' not in x['Name']])
    test_artifact_name_lowercase = Counter([x['Name']==x['Name'].lower() for x in test_data if x['Valid']=='Yes' and 'N/A' not in x['Name']])

    artifact_name_lowercase_balance_score = calculate_imbalance(train_artifact_name_lowercase, dev_artifact_name_lowercase, test_artifact_name_lowercase)

    # Calculate the mean imbalance score for artifact name and artifact name lowercase
    artifact_name_final_balance_score = np.mean([artifact_name_balance_score, artifact_name_lowercase_balance_score])

    train_artifact_version = Counter(['N/A' not in x['Version'] for x in train_data if x['Valid']=='Yes'])
    dev_artifact_version = Counter(['N/A' not in x['Version'] for x in dev_data if x['Valid']=='Yes'])
    test_artifact_version = Counter(['N/A' not in x['Version'] for x in test_data if x['Valid']=='Yes'])

    artifact_version_balance_score = calculate_imbalance(train_artifact_version, dev_artifact_version, test_artifact_version)

    train_artifact_license = Counter(['N/A' not in x['License'] for x in train_data if x['Valid']=='Yes'])
    dev_artifact_license = Counter(['N/A' not in x['License'] for x in dev_data if x['Valid']=='Yes'])
    test_artifact_license = Counter(['N/A' not in x['License'] for x in test_data if x['Valid']=='Yes'])

    artifact_license_balance_score = calculate_imbalance(train_artifact_license, dev_artifact_license, test_artifact_license)

    train_artifact_url = Counter(['N/A' not in x['URL'] for x in train_data if x['Valid']=='Yes'])
    dev_artifact_url = Counter(['N/A' not in x['URL'] for x in dev_data if x['Valid']=='Yes'])
    test_artifact_url = Counter(['N/A' not in x['URL'] for x in test_data if x['Valid']=='Yes'])

    artifact_url_balance_score = calculate_imbalance(train_artifact_url, dev_artifact_url, test_artifact_url)

    train_artifact_ownership = Counter([x['Ownership'] for x in train_data if x['Valid']=='Yes'])
    dev_artifact_ownership = Counter([x['Ownership'] for x in dev_data if x['Valid']=='Yes'])
    test_artifact_ownership = Counter([x['Ownership'] for x in test_data if x['Valid']=='Yes'])

    artifact_ownership_balance_score = calculate_imbalance(train_artifact_ownership, dev_artifact_ownership, test_artifact_ownership)

    train_artifact_usage = Counter([x['Usage'] for x in train_data if x['Valid']=='Yes'])
    dev_artifact_usage = Counter([x['Usage'] for x in dev_data if x['Valid']=='Yes'])
    test_artifact_usage = Counter([x['Usage'] for x in test_data if x['Valid']=='Yes'])

    artifact_usage_balance_score = calculate_imbalance(train_artifact_usage, dev_artifact_usage, test_artifact_usage)

    # Calculate the overall metadata balance score
    artifact_metadata_balance_score = np.mean([artifact_name_final_balance_score, artifact_version_balance_score, artifact_license_balance_score, artifact_url_balance_score, artifact_ownership_balance_score, artifact_usage_balance_score])

    """ OVERALL """
    # Calculate the overall balance score
    balance_score = np.mean([artifact_types_balance_score, artifact_count_balance_score, artifact_metadata_balance_score])

    return balance_score, (train_data, dev_data, test_data)


# Do many iterations to find the best split
best_score = float('inf')
best_split_data = None
best_split_snippets = None
for i in tqdm(range(10000)):
    # Perform a random split of the data to train, dev and test sets (80-10-10)
    random.shuffle(all_unique_snippets)
    train_snippets = all_unique_snippets[:int(len(all_unique_snippets)*0.8)]
    dev_snippets = all_unique_snippets[int(len(all_unique_snippets)*0.8):int(len(all_unique_snippets)*0.9)]
    test_snippets = all_unique_snippets[int(len(all_unique_snippets)*0.9):]

    # Check the score of the split
    score, split = score_split(train_snippets, dev_snippets, test_snippets)

    # If the score is better than the previous best, then save it
    if score < best_score:
        best_score = score
        best_split_data = split
        best_split_snippets = (train_snippets, dev_snippets, test_snippets)
    

# Statistics for the best split
print('Best split score: {}'.format(best_score))
print()
for _data, _snippets, _set in zip(best_split_data, best_split_snippets, ['Train', 'Dev', 'Test']):
    print('# {} data size: {}'.format(_set, len(_data)))

    # Check the number of instances per valid artifact count
    print('> Instances per valid artifact count:')
    instances_per_artifact_count = Counter([len([x2 for x2 in unique_snippets_to_data[x] if x2['Valid']=='Yes']) for x in _snippets])
    for k, v in instances_per_artifact_count.items():
        print('{}: {}'.format(k, v))

    # Check the number of instances per artifact type
    print('> Instances per artifact type:')
    instances_per_artifact_type = Counter([x['Type'] for x in _data])
    for k, v in instances_per_artifact_type.items():
        print('{}: {}'.format(k, v))

    # Check the number of instances per artifact metadata
    print('> Instances per artifact metadata:')
    print('>> Name:')
    instances_per_artifact_metadata_name = Counter(['N/A' not in x['Name'] for x in _data if x['Valid']=='Yes'])
    for k, v in instances_per_artifact_metadata_name.items():
        print('{}: {}'.format(k, v))
    print('>> Name (lowercase):')
    instances_per_artifact_metadata_name_lowercase = Counter([x['Name']==x['Name'].lower() for x in _data if x['Valid']=='Yes' and 'N/A' not in x['Name']])
    for k, v in instances_per_artifact_metadata_name_lowercase.items():
        print('{}: {}'.format(k, v))
    print('>> Version:')
    instances_per_artifact_metadata_version = Counter(['N/A' not in x['Version'] for x in _data if x['Valid']=='Yes'])
    for k, v in instances_per_artifact_metadata_version.items():
        print('{}: {}'.format(k, v))
    print('>> License:')
    instances_per_artifact_metadata_license = Counter(['N/A' not in x['License'] for x in _data if x['Valid']=='Yes'])
    for k, v in instances_per_artifact_metadata_license.items():
        print('{}: {}'.format(k, v))
    print('>> URL:')
    instances_per_artifact_metadata_url = Counter(['N/A' not in x['URL'] for x in _data if x['Valid']=='Yes'])
    for k, v in instances_per_artifact_metadata_url.items():
        print('{}: {}'.format(k, v))
    print('>> Ownership:')
    instances_per_artifact_metadata_ownership = Counter([x['Ownership'] for x in _data if x['Valid']=='Yes'])
    for k, v in instances_per_artifact_metadata_ownership.items():
        print('{}: {}'.format(k, v))
    print('>> Usage:')
    instances_per_artifact_metadata_usage = Counter([x['Usage'] for x in _data if x['Valid']=='Yes'])
    for k, v in instances_per_artifact_metadata_usage.items():
        print('{}: {}'.format(k, v))

    print()

for data, unique_snippets, set_name in zip(best_split_data, best_split_snippets, ['train', 'dev', 'test']):

    print('Creating and augmenting the {} data...'.format(set_name))

    """ SPECIAL-TYPE INSTANCES """

    with open(f'./data/raa_synthetic_dataset_unique_snippets_{set_name}.txt', 'w', encoding='utf-8') as out:
        for snippet in unique_snippets:
            out.write(snippet + '\n')

    # Create a dictionary with the instances per snippet
    snippets_to_instances = {}
    for snippet in unique_snippets:
        snippets_to_instances[snippet] = []
        for d in data:
            if d['Snippet'].replace('<m>', '').replace('</m>', '') == snippet:
                snippets_to_instances[snippet].append(d)

    # Check the number of instances per valid artifact count
    # (Checked and we have a good amount for the valid artifact counts (even for 0))
    instances_per_artifact_count = Counter([len([x2 for x2 in snippets_to_instances[x] if x2['Valid']=='Yes']) for x in snippets_to_instances])

    # Create a list of the valid artifacts per snippet
    snippets_to_valid_artifacts = {}
    for snippet in unique_snippets:
        snippets_to_valid_artifacts[snippet] = []
        for d in snippets_to_instances[snippet]:
            if d['Valid'] == 'Yes':
                if '|' in d['Name']:
                    for name in d['Name'].split('|'):
                        snippets_to_valid_artifacts[snippet].append((d['Type'], name.strip() if 'N/A' not in name else 'unnamed'))
                else:
                    snippets_to_valid_artifacts[snippet].append((d['Type'], d['Name'] if 'N/A' not in d['Name'] else 'unnamed'))
        if len(snippets_to_valid_artifacts[snippet]) > 0:
            snippets_to_valid_artifacts[snippet] = create_ordered_set(snippets_to_valid_artifacts[snippet])

    # Create the special-type instances
    fast_mode_instances = []
    for snippet in snippets_to_valid_artifacts:
        if snippets_to_valid_artifacts[snippet]:
            output_text = ''
            for i, artifact in enumerate(snippets_to_valid_artifacts[snippet]):
                output_text += '{} : {} | '.format(artifact[0], artifact[1])
            if output_text.endswith(' | '):
                output_text = output_text[:-3]
        else:
            output_text = 'N/A'
        fast_mode_instances.append({
            'input': '### Snippet: {} ### Question: List all the artifacts in the above snippet. ### Answer:'.format(snippet),
            'output': output_text
        })


    """ PARAPHRASING """
    from paraphrase_model import paraphrase

    # Paraphrase each instance
    k = 3
    paraphrased_instances = {}
    for d in tqdm(data):
        # Create the paraphrased instances replacing the artifact trigger with the [MASK] token
        paraphrase_snippets = paraphrase(re.sub(r'<m>.*?</m>', '[MASK]', d['Snippet']))

        # Filter the paraphrased instances (remove the duplicates -- keep the shortest to prevent halucinations in the end) 
        # TODO: maybe also check for noise like many dots and questionsmarks, etc
        to_remove = []
        for x in paraphrase_snippets:
            for x2 in paraphrase_snippets:
                if x != x2 and x in x2:
                    to_remove.append(x2)
        to_remove = set(to_remove)
        paraphrase_snippets = create_ordered_set([x for x in paraphrase_snippets if x not in to_remove])

        # Check if the artifact (the [MASK] token) is still present (also check that there is only one [MASK] token)
        paraphrase_snippets = [x for x in paraphrase_snippets if '[MASK]' in x and x.count('[MASK]') == 1]

        # Replace [MASK] token with the artifact trigger
        artifact_trigger = re.findall(r'<m>(.*?)</m>', d['Snippet'])[0]
        paraphrase_snippets = [x.replace('[MASK]', '<m>{}</m>'.format(artifact_trigger)) for x in paraphrase_snippets]

        # Remove the paraphrased snippets that are the same as the original snippet
        paraphrase_snippets = [x for x in paraphrase_snippets if x != d['Snippet']][:k]

        # Add the paraphrased snippets to the dictionary
        paraphrased_instances[d['Snippet']] = paraphrase_snippets

    # Create the paraphrase instances data
    paraphrased_instances_data = []
    for snippet in paraphrased_instances:
        for paraphrase_snippet in paraphrased_instances[snippet]:
            data_instance = deepcopy(all_data_dict[snippet])
            data_instance['Snippet'] = paraphrase_snippet
            paraphrased_instances_data.append(data_instance)

    # Paraphrase Special-Type instances
    k = 3
    paraphrase_snippets_to_valid_artifacts = {}
    for snippet in tqdm(unique_snippets):
        # Create the paraphrased instances
        paraphrase_snippets = paraphrase(snippet)

        # Filter the paraphrased instances (remove the duplicates -- keep the shortest to prevent halucinations in the end) 
        # TODO: maybe also check for noise like many dots and questionsmarks, etc
        to_remove = []
        for x in paraphrase_snippets:
            for x2 in paraphrase_snippets:
                if x != x2 and x in x2:
                    to_remove.append(x2)
        to_remove = set(to_remove)
        paraphrase_snippets = create_ordered_set([x for x in paraphrase_snippets if x not in to_remove])

        # Remove the paraphrased snippets that are the same as the original snippet
        paraphrase_snippets = [x for x in paraphrase_snippets if x != d['Snippet']]

        # Check how many artifacts are in the paraphrased snippets
        paraphrase_snippets_included = 0
        valid_artifacts = snippets_to_valid_artifacts[snippet]
        if valid_artifacts:
            for x in paraphrase_snippets:
                # Bool used to discard the snippet if it doesn't contain all the artifacts
                discard_snippet = False
                valid_artifacts_in_paraphrase = []
                for artifact in valid_artifacts:
                    if artifact[1] == 'unnamed' or artifact[1] in x:
                        valid_artifacts_in_paraphrase.append(artifact)
                    else:
                        # If the artifact is not in the paraphrase, try to first find in lowercase and extract it's original case
                        _res = list(re.finditer(re.escape(artifact[1]), x, re.IGNORECASE))
                        if _res:
                            _res = _res[0]
                            _artifact = x[_res.start():_res.end()]
                            valid_artifacts_in_paraphrase.append((artifact[0], _artifact))
                        
                        # If not then discard the snippet
                        else:
                            discard_snippet = True
                            break
                if not discard_snippet:
                    paraphrase_snippets_to_valid_artifacts[x] = valid_artifacts_in_paraphrase
                    paraphrase_snippets_included += 1

                if paraphrase_snippets_included == k:
                    break
        else:
            for x in paraphrase_snippets:
                paraphrase_snippets_to_valid_artifacts[x] = []

    # Create the paraphrase special-type instances
    paraphrase_fast_mode_instances = []
    for snippet in paraphrase_snippets_to_valid_artifacts:
        if paraphrase_snippets_to_valid_artifacts[snippet]:
            output_text = ''
            for i, artifact in enumerate(paraphrase_snippets_to_valid_artifacts[snippet]):
                output_text += '{} : {} | '.format(artifact[0], artifact[1])
            if output_text.endswith(' | '):
                output_text = output_text[:-3]
        else:
            output_text = 'N/A'
        paraphrase_fast_mode_instances.append({
            'input': '### Snippet: {} ### Question: List all the artifacts in the above snippet. ### Answer:'.format(snippet),
            'output': output_text
        })
    

    """ LAST PART -- ADD THEM TO THE TRANSFORMED DATA """

    
    question_per_field = {
        "Valid": "Is there a valid {} defined in the <m> and </m> tags?",
        "Name": "What is the name of the {} defined in the <m> and </m> tags?",
        "Version": "What is the version of the {} defined in the <m> and </m> tags?",
        "License": "What is the license of the {} defined in the <m> and </m> tags?",
        "URL": "What is the URL of the {} defined in the <m> and </m> tags?",
        "Ownership": "Is the {} defined in the <m> and </m> tags introduced or created by the authors of the publication in the snippet above?",
        "Usage": "Is the {} defined in the <m> and </m> tags used or adopted by the authors of the publication in the snippet above?"
    }

    transformed_data = []
    for d in data:
        for field in question_per_field:
            if field in d:
                transformed_data.append({
                    "input": '### Snippet: {} ### Question: {} ### Answer:'.format(d['Snippet'], question_per_field[field].format(d['Type'])),
                    "output": d[field]
                })

    for d in paraphrased_instances_data:
        for field in question_per_field:
            if field in d:
                transformed_data.append({
                    "input": '### Snippet: {} ### Question: {} ### Answer:'.format(d['Snippet'], question_per_field[field].format(d['Type'])),
                    "output": d[field]
                })

    # Special-Type Instances
    transformed_data.extend(fast_mode_instances)
    transformed_data.extend(paraphrase_fast_mode_instances)

    with open(f'./data/raa_synthetic_dataset_aug_transformed_{set_name}.json', 'w', encoding='utf-8') as out:
        json.dump(transformed_data, out, indent=1)
