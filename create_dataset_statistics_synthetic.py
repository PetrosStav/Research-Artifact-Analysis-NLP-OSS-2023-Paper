import json
import pandas as pd

# MAP TYPES
type_mapping = {
    'method': 'software',
    'repository': 'dataset',
    'dataset': 'dataset',
    'software': 'software',
}

question_part_to_field = {
    "Valid": "Is there a valid",
    "Name": "What is the name of the ",
    "Version": "What is the version of the ",
    "License": "What is the license of the ",
    "URL": "What is the URL of the ",
    "Ownership": " defined in the <m> and </m> tags introduced or created by the authors of the publication in the snippet above?",
    "Usage": " defined in the <m> and </m> tags used or adopted by the authors of the publication in the snippet above?",
    'Special-Type': "List all"
}

question_per_field = {
    "Valid": "Is there a valid {} defined in the <m> and </m> tags?",
    "Name": "What is the name of the {} defined in the <m> and </m> tags?",
    "Version": "What is the version of the {} defined in the <m> and </m> tags?",
    "License": "What is the license of the {} defined in the <m> and </m> tags?",
    "URL": "What is the URL of the {} defined in the <m> and </m> tags?",
    "Ownership": "Is the {} defined in the <m> and </m> tags introduced or created by the authors of the publication in the snippet above?",
    "Usage": "Is the {} defined in the <m> and </m> tags used or adopted by the authors of the publication in the snippet above?"
}

question_per_field_rev = {question_per_field[k].format('dataset'):[k, 'dataset'] for k in question_per_field}
question_per_field_rev.update({question_per_field[k].format('software'):[k, 'software'] for k in question_per_field})

with open('./data/raa_synthetic_dataset_aug_transformed_train.json', 'r') as f:
    train_data = json.load(f)

with open('./data/raa_synthetic_dataset_aug_transformed_dev.json', 'r') as f:
    dev_data = json.load(f)

with open('./data/raa_synthetic_dataset_aug_transformed_test.json', 'r') as f:
    test_data = json.load(f)

og_stats = dict()
aug_stats = dict()

""" ORIGINAL - SNIPPETS """

with open('./data/raa_synthetic_dataset.json', 'r') as f:
    og_data = json.load(f)

for d in og_data:
    d['Type'] = type_mapping[d['Type']]

with open('./data/raa_synthetic_dataset_unique_snippets_train.txt', 'r') as f:
    og_train_snippets = f.read().splitlines()

og_train_snippets_lower = [s.lower() for s in og_train_snippets]

with open('./data/raa_synthetic_dataset_unique_snippets_dev.txt', 'r') as f:
    og_dev_snippets = f.read().splitlines()

og_dev_snippets_lower = [s.lower() for s in og_dev_snippets]

with open('./data/raa_synthetic_dataset_unique_snippets_test.txt', 'r') as f:
    og_test_snippets = f.read().splitlines()

og_test_snippets_lower = [s.lower() for s in og_test_snippets]

# Split the og data into train, dev and test
og_train_data = [d for d in og_data if d['Snippet'].replace('<m>', '').replace('</m>', '').lower() in og_train_snippets_lower]
og_dev_data = [d for d in og_data if d['Snippet'].replace('<m>', '').replace('</m>', '').lower() in og_dev_snippets_lower]
og_test_data = [d for d in og_data if d['Snippet'].replace('<m>', '').replace('</m>', '').lower() in og_test_snippets_lower]

og_stats['snippets'] = dict()
for _set, _set_name in zip([og_train_data, og_dev_data, og_test_data], ['Train', 'Dev', 'Test']):

    # Get unique snippets (after removing <m> and </m> tags)
    unique_snippets = sorted({s['Snippet'].replace('<m>', '').replace('</m>', '') for s in _set})

    # Get statistics for each artifact type and for the whole dataset
    artifact_types = sorted({s['Type'] for s in _set})

    og_stats['snippets'][_set_name] = dict()
    for a_type in artifact_types:
        og_stats['snippets'][_set_name][a_type] = {
            'Number of instances': len({s['Snippet'] for s in _set if s['Type']==a_type}),
            'Number valid': len({s['Snippet'] for s in _set if s['Type']==a_type and 'Valid' in s and s['Valid']=='Yes'}),
            'Number w. name': len({s['Snippet'] for s in _set if s['Type']==a_type and 'Name' in s and 'N/A' not in s['Name']}),
            'Number w. version': len({s['Snippet'] for s in _set if s['Type']==a_type and 'Version' in s and 'N/A' not in s['Version']}),
            'Number w. license': len({s['Snippet'] for s in _set if s['Type']==a_type and 'License' in s and 'N/A' not in s['License']}),
            'Number w. URL': len({s['Snippet'] for s in _set if s['Type']==a_type and 'URL' in s and 'N/A' not in s['URL']}),
            'Number w. ownership': len({s['Snippet'] for s in _set if s['Type']==a_type and 'Ownership' in s and 'Yes' in s['Ownership']}),
            'Number w. usage': len({s['Snippet'] for s in _set if s['Type']==a_type and 'Usage' in s and 'Yes' in s['Usage']}),
            'Number of unique snippets': len({s['Snippet'].replace('<m>', '').replace('</m>', '') for s in _set if s['Type']==a_type}),
        }

        # Add percentages
        for k in og_stats['snippets'][_set_name][a_type]:
            if k != 'Number of instances' and k != 'Number of unique snippets':
                og_stats['snippets'][_set_name][a_type][k] = [og_stats['snippets'][_set_name][a_type][k], round(og_stats['snippets'][_set_name][a_type][k] / og_stats['snippets'][_set_name][a_type]['Number of instances'] * 100, 2)]

    og_stats['snippets'][_set_name]['all'] = {
        'Number of instances': len({s['Snippet'] for s in _set}),
        'Number valid': len({s['Snippet'] for s in _set if 'Valid' in s and s['Valid']=='Yes'}),
        'Number w. name': len({s['Snippet'] for s in _set if 'Name' in s and 'N/A' not in s['Name']}),
        'Number w. version': len({s['Snippet'] for s in _set if 'Version' in s and 'N/A' not in s['Version']}),
        'Number w. license': len({s['Snippet'] for s in _set if 'License' in s and 'N/A' not in s['License']}),
        'Number w. URL': len({s['Snippet'] for s in _set if 'URL' in s and 'N/A' not in s['URL']}),
        'Number w. ownership': len({s['Snippet'] for s in _set if 'Ownership' in s and 'Yes' in s['Ownership']}),
        'Number w. usage': len({s['Snippet'] for s in _set if 'Usage' in s and 'Yes' in s['Usage']}),
        'Number of unique snippets': len(unique_snippets),
    }

    # Add percentages
    for k in og_stats['snippets'][_set_name]['all']:
        if k != 'Number of instances' and k != 'Number of unique snippets':
            og_stats['snippets'][_set_name]['all'][k] = [og_stats['snippets'][_set_name]['all'][k], round(og_stats['snippets'][_set_name]['all'][k] / og_stats['snippets'][_set_name]['all']['Number of instances'] * 100, 2)]

""" ORIGINAL - QUESTION-ANSWER PAIRS """

transformed_data = []
for data in [og_train_data, og_dev_data, og_test_data]:
    transformed_data.append(list())
    for d in data:
        for field in question_per_field:
            if field in d:
                transformed_data[-1].append({
                    "input": '### Snippet: {} ### Question: {} ### Answer:'.format(d['Snippet'], question_per_field[field].format(d['Type'])),
                    "output": d[field]
                })

og_stats['qa_pairs'] = dict()
num_instances_per_type = dict()
for _set, _set_name in zip([transformed_data[0], transformed_data[1], transformed_data[2]], ['Train', 'Dev', 'Test']):
    num_instances_per_type[_set_name] = dict()
    # Find snippets
    snippets_dict = dict()
    fast_mode_dict = dict()
    for d in _set:
        snippet = d['input'].split('### Question:')[0].split('### Snippet:')[1].strip()
        question = d['input'].split('### Question:')[1].split('### Answer:')[0].strip()
        for k in question_part_to_field:
            if question_part_to_field[k] in question:
                if k == 'Special-Type':
                    if snippet not in fast_mode_dict:
                        fast_mode_dict[snippet] = d['output']
                    else:
                        print('ERROR:', _set_name)
                        print('snippet:', snippet)
                else:
                    if snippet not in snippets_dict:
                        snippets_dict[snippet] = {
                            'Snippet': snippet,
                        }
                    
                    # Find artifact type
                    if k in ['Ownership', 'Usage']:
                        artifact_type = question.split(question_part_to_field[k])[0].split('Is the ')[1].strip()
                    else:
                        artifact_type = question.split(question_part_to_field[k])[1].split(' defined in the <m> and </m> tags?')[0].strip()
                    
                    artifact_type = type_mapping[artifact_type]

                    snippets_dict[snippet]['Type'] = artifact_type
                    num_instances_per_type[_set_name][artifact_type] = num_instances_per_type[_set_name][artifact_type] + 1 if artifact_type in num_instances_per_type[_set_name] else 1
                    snippets_dict[snippet][k] = d['output']
                break

    # Get unique snippets (after removing <m> and </m> tags)
    unique_snippets = sorted({s.replace('<m>', '').replace('</m>', '') for s in snippets_dict})

    # Get statistics for each artifact type and for the whole dataset
    artifact_types = sorted({snippets_dict[s]['Type'] for s in snippets_dict})

    og_stats['qa_pairs'][_set_name] = dict()
    for a_type in artifact_types:
        og_stats['qa_pairs'][_set_name][a_type] = {
            'Number of instances': num_instances_per_type[_set_name][a_type],
            'Number of snippets w. tags': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type]),
            'Number valid': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'Valid' in snippets_dict[s] and snippets_dict[s]['Valid']=='Yes']),
            'Number w. name': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'Name' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['Name']]),
            'Number w. version': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'Version' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['Version']]),
            'Number w. license': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'License' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['License']]),
            'Number w. URL': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'URL' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['URL']]),
            'Number w. ownership': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'Ownership' in snippets_dict[s] and 'Yes' in snippets_dict[s]['Ownership']]),
            'Number w. usage': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'Usage' in snippets_dict[s] and 'Yes' in snippets_dict[s]['Usage']]),
            'Number of unique snippets': len({s for s in snippets_dict if snippets_dict[s]['Type']==a_type}),
            'Number of special-type': len([s for s in fast_mode_dict if any(x.split(':')[0].strip()==a_type for x in fast_mode_dict[s].split('|'))]),
        }

        # Add percentages
        for k in og_stats['qa_pairs'][_set_name][a_type]:
            if k != 'Number of instances' and k != 'Number of unique snippets' and k != 'Number of special-type' and k != 'Number of snippets w. tags':
                og_stats['qa_pairs'][_set_name][a_type][k] = [og_stats['qa_pairs'][_set_name][a_type][k], round(og_stats['qa_pairs'][_set_name][a_type][k] / og_stats['qa_pairs'][_set_name][a_type]['Number of snippets w. tags'] * 100, 2)]

    og_stats['qa_pairs'][_set_name]['all'] = {
        'Number of instances': len(_set),
        'Number of snippets w. tags': len(snippets_dict),
        'Number valid': len([s for s in snippets_dict if 'Valid' in snippets_dict[s] and snippets_dict[s]['Valid']=='Yes']),
        'Number w. name': len([s for s in snippets_dict if 'Name' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['Name']]),
        'Number w. version': len([s for s in snippets_dict if 'Version' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['Version']]),
        'Number w. license': len([s for s in snippets_dict if 'License' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['License']]),
        'Number w. URL': len([s for s in snippets_dict if 'URL' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['URL']]),
        'Number w. ownership': len([s for s in snippets_dict if 'Ownership' in snippets_dict[s] and 'Yes' in snippets_dict[s]['Ownership']]),
        'Number w. usage': len([s for s in snippets_dict if 'Usage' in snippets_dict[s] and 'Yes' in snippets_dict[s]['Usage']]),
        'Number of unique snippets': len(unique_snippets),
        'Number of special-type': len(fast_mode_dict)
    }

    # Add percentages
    for k in og_stats['qa_pairs'][_set_name]['all']:
        if k != 'Number of instances' and k != 'Number of unique snippets' and k != 'Number of special-type' and k != 'Number of snippets w. tags':
            og_stats['qa_pairs'][_set_name]['all'][k] = [og_stats['qa_pairs'][_set_name]['all'][k], round(og_stats['qa_pairs'][_set_name]['all'][k] / og_stats['qa_pairs'][_set_name]['all']['Number of snippets w. tags'] * 100, 2)]

""" AGGREGATE - SNIPPETS """

# Find the snippets in the augmented dataset and convert them into the original format
aug_data_snippets = []
for data in [train_data, dev_data, test_data]:
    aug_data_snippets.append(dict())
    for d in data:
        snippet = d['input'].split('### Question:')[0].split('### Snippet:')[1].strip()
        question = d['input'].split('### Question:')[1].split('### Answer:')[0].strip()
        if question == 'List all the artifacts in the above snippet.':
            continue
        answer = d['output']
        type = question_per_field_rev[question][1]
        field = question_per_field_rev[question][0]
        if snippet not in aug_data_snippets[-1]:
            aug_data_snippets[-1][snippet] = {
                'Snippet': snippet,
                'Type': type,
            }
        aug_data_snippets[-1][snippet][field] = answer

# Split the og data into train, dev and test
aug_train_data = list(aug_data_snippets[0].values())
aug_dev_data = list(aug_data_snippets[1].values())
aug_test_data = list(aug_data_snippets[2].values())

aug_stats['snippets'] = dict()
for _set, _set_name in zip([aug_train_data, aug_dev_data, aug_test_data], ['Train', 'Dev', 'Test']):

    # Get unique snippets (after removing <m> and </m> tags)
    unique_snippets = sorted({s['Snippet'].replace('<m>', '').replace('</m>', '') for s in _set})

    # Get statistics for each artifact type and for the whole dataset
    artifact_types = sorted({s['Type'] for s in _set})

    aug_stats['snippets'][_set_name] = dict()
    for a_type in artifact_types:
        aug_stats['snippets'][_set_name][a_type] = {
            'Number of instances': len({s['Snippet'] for s in _set if s['Type']==a_type}),
            'Number valid': len({s['Snippet'] for s in _set if s['Type']==a_type and 'Valid' in s and s['Valid']=='Yes'}),
            'Number w. name': len({s['Snippet'] for s in _set if s['Type']==a_type and 'Name' in s and 'N/A' not in s['Name']}),
            'Number w. version': len({s['Snippet'] for s in _set if s['Type']==a_type and 'Version' in s and 'N/A' not in s['Version']}),
            'Number w. license': len({s['Snippet'] for s in _set if s['Type']==a_type and 'License' in s and 'N/A' not in s['License']}),
            'Number w. URL': len({s['Snippet'] for s in _set if s['Type']==a_type and 'URL' in s and 'N/A' not in s['URL']}),
            'Number w. ownership': len({s['Snippet'] for s in _set if s['Type']==a_type and 'Ownership' in s and 'Yes' in s['Ownership']}),
            'Number w. usage': len({s['Snippet'] for s in _set if s['Type']==a_type and 'Usage' in s and 'Yes' in s['Usage']}),
            'Number of unique snippets': len({s['Snippet'].replace('<m>', '').replace('</m>', '') for s in _set if s['Type']==a_type}),
        }

        # Add percentages
        for k in aug_stats['snippets'][_set_name][a_type]:
            if k != 'Number of instances' and k != 'Number of unique snippets':
                aug_stats['snippets'][_set_name][a_type][k] = [aug_stats['snippets'][_set_name][a_type][k], round(aug_stats['snippets'][_set_name][a_type][k] / aug_stats['snippets'][_set_name][a_type]['Number of instances'] * 100, 2)]

    aug_stats['snippets'][_set_name]['all'] = {
        'Number of instances': len({s['Snippet'] for s in _set}),
        'Number valid': len({s['Snippet'] for s in _set if 'Valid' in s and s['Valid']=='Yes'}),
        'Number w. name': len({s['Snippet'] for s in _set if 'Name' in s and 'N/A' not in s['Name']}),
        'Number w. version': len({s['Snippet'] for s in _set if 'Version' in s and 'N/A' not in s['Version']}),
        'Number w. license': len({s['Snippet'] for s in _set if 'License' in s and 'N/A' not in s['License']}),
        'Number w. URL': len({s['Snippet'] for s in _set if 'URL' in s and 'N/A' not in s['URL']}),
        'Number w. ownership': len({s['Snippet'] for s in _set if 'Ownership' in s and 'Yes' in s['Ownership']}),
        'Number w. usage': len({s['Snippet'] for s in _set if 'Usage' in s and 'Yes' in s['Usage']}),
        'Number of unique snippets': len(unique_snippets),
    }

    # Add percentages
    for k in aug_stats['snippets'][_set_name]['all']:
        if k != 'Number of instances' and k != 'Number of unique snippets':
            aug_stats['snippets'][_set_name]['all'][k] = [aug_stats['snippets'][_set_name]['all'][k], round(aug_stats['snippets'][_set_name]['all'][k] / aug_stats['snippets'][_set_name]['all']['Number of instances'] * 100, 2)]


""" AUGMENTED - QUESTION-ANSWER PAIRS """

aug_stats['qa_pairs'] = dict()
num_instances_per_type = dict()
for _set, _set_name in zip([train_data, dev_data, test_data], ['Train', 'Dev', 'Test']):
    num_instances_per_type[_set_name] = dict()
    # Find snippets
    snippets_dict = dict()
    fast_mode_dict = dict()
    for d in _set:
        snippet = d['input'].split('### Question:')[0].split('### Snippet:')[1].strip()
        question = d['input'].split('### Question:')[1].split('### Answer:')[0].strip()
        for k in question_part_to_field:
            if question_part_to_field[k] in question:
                if k == 'Special-Type':
                    if snippet not in fast_mode_dict:
                        fast_mode_dict[snippet] = d['output']
                    else:
                        print('ERROR:', _set_name)
                        print('snippet:', snippet)
                else:
                    if snippet not in snippets_dict:
                        snippets_dict[snippet] = {
                            'Snippet': snippet,
                        }
                    
                    # Find artifact type
                    if k in ['Ownership', 'Usage']:
                        artifact_type = question.split(question_part_to_field[k])[0].split('Is the ')[1].strip()
                    else:
                        artifact_type = question.split(question_part_to_field[k])[1].split(' defined in the <m> and </m> tags?')[0].strip()
                    
                    artifact_type = type_mapping[artifact_type]

                    snippets_dict[snippet]['Type'] = artifact_type
                    num_instances_per_type[_set_name][artifact_type] = num_instances_per_type[_set_name][artifact_type] + 1 if artifact_type in num_instances_per_type[_set_name] else 1
                    snippets_dict[snippet][k] = d['output']
                break

    # Get unique snippets (after removing <m> and </m> tags)
    unique_snippets = sorted({s.replace('<m>', '').replace('</m>', '') for s in snippets_dict})

    # Get statistics for each artifact type and for the whole dataset
    artifact_types = sorted({snippets_dict[s]['Type'] for s in snippets_dict})

    aug_stats['qa_pairs'][_set_name] = dict()
    for a_type in artifact_types:
        aug_stats['qa_pairs'][_set_name][a_type] = {
            'Number of instances': num_instances_per_type[_set_name][a_type],
            'Number of snippets w. tags': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type]),
            'Number valid': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'Valid' in snippets_dict[s] and snippets_dict[s]['Valid']=='Yes']),
            'Number w. name': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'Name' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['Name']]),
            'Number w. version': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'Version' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['Version']]),
            'Number w. license': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'License' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['License']]),
            'Number w. URL': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'URL' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['URL']]),
            'Number w. ownership': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'Ownership' in snippets_dict[s] and 'Yes' in snippets_dict[s]['Ownership']]),
            'Number w. usage': len([s for s in snippets_dict if snippets_dict[s]['Type']==a_type and 'Usage' in snippets_dict[s] and 'Yes' in snippets_dict[s]['Usage']]),
            'Number of unique snippets': len({s for s in snippets_dict if snippets_dict[s]['Type']==a_type}),
            'Number of special-type': len([s for s in fast_mode_dict if any(x.split(':')[0].strip()==a_type for x in fast_mode_dict[s].split('|'))]),
        }

        # Add percentages
        for k in aug_stats['qa_pairs'][_set_name][a_type]:
            if k != 'Number of instances' and k != 'Number of unique snippets' and k != 'Number of special-type' and k != 'Number of snippets w. tags':
                aug_stats['qa_pairs'][_set_name][a_type][k] = [aug_stats['qa_pairs'][_set_name][a_type][k], round(aug_stats['qa_pairs'][_set_name][a_type][k] / aug_stats['qa_pairs'][_set_name][a_type]['Number of snippets w. tags'] * 100, 2)]

    aug_stats['qa_pairs'][_set_name]['all'] = {
        'Number of instances': len(_set),
        'Number of snippets w. tags': len(snippets_dict),
        'Number valid': len([s for s in snippets_dict if 'Valid' in snippets_dict[s] and snippets_dict[s]['Valid']=='Yes']),
        'Number w. name': len([s for s in snippets_dict if 'Name' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['Name']]),
        'Number w. version': len([s for s in snippets_dict if 'Version' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['Version']]),
        'Number w. license': len([s for s in snippets_dict if 'License' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['License']]),
        'Number w. URL': len([s for s in snippets_dict if 'URL' in snippets_dict[s] and 'N/A' not in snippets_dict[s]['URL']]),
        'Number w. ownership': len([s for s in snippets_dict if 'Ownership' in snippets_dict[s] and 'Yes' in snippets_dict[s]['Ownership']]),
        'Number w. usage': len([s for s in snippets_dict if 'Usage' in snippets_dict[s] and 'Yes' in snippets_dict[s]['Usage']]),
        'Number of unique snippets': len(unique_snippets),
        'Number of special-type': len(fast_mode_dict)
    }

    # Add percentages
    for k in aug_stats['qa_pairs'][_set_name]['all']:
        if k != 'Number of instances' and k != 'Number of unique snippets' and k != 'Number of special-type' and k != 'Number of snippets w. tags':
            aug_stats['qa_pairs'][_set_name]['all'][k] = [aug_stats['qa_pairs'][_set_name]['all'][k], round(aug_stats['qa_pairs'][_set_name]['all'][k] / aug_stats['qa_pairs'][_set_name]['all']['Number of snippets w. tags'] * 100, 2)]

""" SAVE ALL """

with open('./material/synthetic_dataset_stats.json', 'w') as f:
    json.dump({
        'original_snippets': og_stats['snippets'],
        'original_qa_pairs': og_stats['qa_pairs'],
        'augmented_snippets': aug_stats['snippets'],
        'augmented_qa_pairs': aug_stats['qa_pairs'],
    }, f, indent=1)


""" CONVERT ORIGINAL """

# Columns and Index for DataFrame
columns = pd.MultiIndex.from_product([['Train', 'Dev', 'Test'], ['dataset', 'software', 'all']])
index = ['Number of instances', 'Number valid', 'Number w. name', 'Number w. version', 'Number w. license', 'Number w. URL', 'Number w. ownership', 'Number w. usage', 'Number of unique snippets']

# Create empty DataFrame
df = pd.DataFrame(index=index, columns=columns)

# Populate DataFrame with values from JSON
for train_dev_test, values in og_stats['snippets'].items():
    for category, category_values in values.items():
        for key in index:
            if isinstance(category_values[key], list):
                # If value is list (count, percentage), convert to "count (percentage)"
                value = f"{category_values[key][0]} ({category_values[key][1]} %)"
            else:
                # If value is count, use as-is
                value = category_values[key]
            # Assign the value to the DataFrame
            df.loc[key, (train_dev_test, category)] = value

# Save DataFrame to Excel
df.to_excel('./material/synthetic_dataset_og_stats_snippets.xlsx')

# Columns and Index for DataFrame
columns = pd.MultiIndex.from_product([['Train', 'Dev', 'Test'], ['dataset', 'software', 'all']])
index = ['Number of instances', 'Number of snippets w. tags', 'Number valid', 'Number w. name', 'Number w. version', 'Number w. license', 'Number w. URL', 'Number w. ownership', 'Number w. usage', 'Number of unique snippets', 'Number of special-type']

# Create empty DataFrame
df = pd.DataFrame(index=index, columns=columns)

# Populate DataFrame with values from JSON
for train_dev_test, values in og_stats['qa_pairs'].items():
    for category, category_values in values.items():
        for key in index:
            if isinstance(category_values[key], list):
                # If value is list (count, percentage), convert to "count (percentage)"
                value = f"{category_values[key][0]} ({category_values[key][1]} %)"
            else:
                # If value is count, use as-is
                value = category_values[key]
            # Assign the value to the DataFrame
            df.loc[key, (train_dev_test, category)] = value

# Save DataFrame to Excel
df.to_excel('./material/synthetic_dataset_og_stats_qa_pairs.xlsx')

""" CONVERT AUGMENTED """

# Columns and Index for DataFrame
columns = pd.MultiIndex.from_product([['Train', 'Dev', 'Test'], ['dataset', 'software', 'all']])
index = ['Number of instances', 'Number valid', 'Number w. name', 'Number w. version', 'Number w. license', 'Number w. URL', 'Number w. ownership', 'Number w. usage', 'Number of unique snippets']

# Create empty DataFrame
df = pd.DataFrame(index=index, columns=columns)

# Populate DataFrame with values from JSON
for train_dev_test, values in aug_stats['snippets'].items():
    for category, category_values in values.items():
        for key in index:
            if isinstance(category_values[key], list):
                # If value is list (count, percentage), convert to "count (percentage)"
                value = f"{category_values[key][0]} ({category_values[key][1]} %)"
            else:
                # If value is count, use as-is
                value = category_values[key]
            # Assign the value to the DataFrame
            df.loc[key, (train_dev_test, category)] = value

# Save DataFrame to Excel
df.to_excel('./material/synthetic_dataset_aug_stats_snippets.xlsx')

# Columns and Index for DataFrame
columns = pd.MultiIndex.from_product([['Train', 'Dev', 'Test'], ['dataset', 'software', 'all']])
index = ['Number of instances', 'Number of snippets w. tags', 'Number valid', 'Number w. name', 'Number w. version', 'Number w. license', 'Number w. URL', 'Number w. ownership', 'Number w. usage', 'Number of unique snippets', 'Number of special-type']

# Create empty DataFrame
df = pd.DataFrame(index=index, columns=columns)

# Populate DataFrame with values from JSON
for train_dev_test, values in aug_stats['qa_pairs'].items():
    for category, category_values in values.items():
        for key in index:
            if isinstance(category_values[key], list):
                # If value is list (count, percentage), convert to "count (percentage)"
                value = f"{category_values[key][0]} ({category_values[key][1]} %)"
            else:
                # If value is count, use as-is
                value = category_values[key]
            # Assign the value to the DataFrame
            df.loc[key, (train_dev_test, category)] = value

# Save DataFrame to Excel
df.to_excel('./material/synthetic_dataset_aug_stats_qa_pairs.xlsx')
