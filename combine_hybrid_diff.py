import json

# Load the data
with open('./data/raa_synthetic_dataset_aug_transformed_train.json') as f:
    synthetic_train_data = json.load(f)

with open('./data/raa_synthetic_dataset_aug_transformed_dev.json') as f:
    synthetic_dev_data = json.load(f)

with open('./data/raa_synthetic_dataset_aug_transformed_test.json') as f:
    synthetic_test_data = json.load(f)

with open('./data/raa_hybrid_dataset_diff_aug_transformed_train.json') as f:
    hybrid_diff_train_data = json.load(f)

with open('./data/raa_hybrid_dataset_diff_aug_transformed_dev.json') as f:
    hybrid_diff_dev_data = json.load(f)

with open('./data/raa_hybrid_dataset_diff_aug_transformed_test.json') as f:
    hybrid_diff_test_data = json.load(f)

# Combine the data
combined_train_data = synthetic_train_data + hybrid_diff_train_data
combined_dev_data = synthetic_dev_data + hybrid_diff_dev_data
combined_test_data = synthetic_test_data + hybrid_diff_test_data

# Save the data
with open('./data/raa_hybrid_dataset_aug_transformed_train.json', 'w') as f:
    json.dump(combined_train_data, f)

with open('./data/raa_hybrid_dataset_aug_transformed_dev.json', 'w') as f:
    json.dump(combined_dev_data, f)

with open('./data/raa_hybrid_dataset_aug_transformed_test.json', 'w') as f:
    json.dump(combined_test_data, f)

# Load the unique snippets
with open('./data/raa_synthetic_dataset_unique_snippets_train.txt') as f:
    synthetic_train_unique_snippets = f.readlines()

with open('./data/raa_synthetic_dataset_unique_snippets_dev.txt') as f:
    synthetic_dev_unique_snippets = f.readlines()

with open('./data/raa_synthetic_dataset_unique_snippets_test.txt') as f:
    synthetic_test_unique_snippets = f.readlines()

with open('./data/raa_hybrid_dataset_diff_unique_snippets_train.txt') as f:
    hybrid_diff_train_unique_snippets = f.readlines()

with open('./data/raa_hybrid_dataset_diff_unique_snippets_dev.txt') as f:
    hybrid_diff_dev_unique_snippets = f.readlines()

with open('./data/raa_hybrid_dataset_diff_unique_snippets_test.txt') as f:
    hybrid_diff_test_unique_snippets = f.readlines()

# Combine the unique snippets
combined_train_unique_snippets = synthetic_train_unique_snippets + hybrid_diff_train_unique_snippets
combined_dev_unique_snippets = synthetic_dev_unique_snippets + hybrid_diff_dev_unique_snippets
combined_test_unique_snippets = synthetic_test_unique_snippets + hybrid_diff_test_unique_snippets

# Save the unique snippets
with open('./data/raa_hybrid_dataset_unique_snippets_train.txt', 'w') as f:
    for snippet in combined_train_unique_snippets:
        f.write(snippet + '\n')

with open('./data/raa_hybrid_dataset_unique_snippets_dev.txt', 'w') as f:
    for snippet in combined_dev_unique_snippets:
        f.write(snippet + '\n')

with open('./data/raa_hybrid_dataset_unique_snippets_test.txt', 'w') as f:
    for snippet in combined_test_unique_snippets:
        f.write(snippet + '\n')
