# Research-Artifact-Analysis-NLP-OSS-2023-Paper
This repository contains the files supporting the paper **"Empowering Knowledge Discovery from Scientific Literature: A novel approach to Research Artifact Analysis"**.

# Task Description

The goal of our paper was to create a gold-annotated dataset and train a simple LLM model (`Flan-T5`) to transform the `Research Artifact Analysis (RAA)` task to an `instruction-based Question Answering (QA)` task.

We created two datasets: `Synthetic` and `Hybrid`, and trained two models: `LoRA-Sy` and `LoRA-Hy` using the [LoRA](https://arxiv.org/pdf/2106.09685.pdf) method.

The datasets contain **named and unnamed (valid) mentions of dataset and software Research Artifacts (RAs)**, as well as **invalid mentions to datasets or software (e.g., general references)**. The **valid RA mentions** also contain the following metadata: **name, version, license, URL, provenance and usage**.

**Snippet example** from the `Hybrid` dataset:

|                     |                                                                                                                                                                                                                      |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Snippet**      | In their study, the authors utilized the PyTorch \<m>library\</m> (version 1.9.0) for deep learning experiments. PyTorch is released under the BSD-3-Clause license. For more information, visit https://pytorch.org/. |
| **Type**       | Software                                                                                                                                                                                                             |
| **Valid**      | Yes                                                                                                                                                                                                                  |
| **Name**       | PyTorch                                                                                                                                                                                                              |
| **Version**    | 1.9.0                                                                                                                                                                                                                |
| **License**    | BSD-3-Clause                                                                                                                                                                                                         |
| **URL**        | https://pytorch.org/                                                                                                                                                                                                 |
| **Provenance** | No                                                                                                                                                                                                                   |
| **Usage**      | Yes                                                                                                                                                                                                                  |

Each snippet in the above datasets are transformed into **Question-Answer (QA) pairs**, so that they can be answered by the LLMs and fill out the above Snippet template.

**QA pair transformation** of the above example:

|                                                                                                                                                  |                            |
|--------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| **Questions**                                                                                                                                 | **Answer**            |
| Is there valid software defined in the \<m> and \</m> tags?                                                                      | Yes                        |
| What is the name of the software defined in the \<m> and \</m> tags?                                                             | PyTorch                    |
| What is the version of the software defined in the \<m> and \</m> tags?                                                          | 1.9.0                      |
| What is the license of the software defined in the \<m> and \</m> tags?                                                          | BSD-3-Clause               |
| What is the URL of the software defined in the \<m> and \</m> tags?                                                              | https://pytorch.org/ |
| Is the software defined in the \<m> and \</m> tags introduced or created by the authors of the publication in the snippet above? | No                         |
| Is the software defined in the \<m> and \</m> tags used or adopted by the authors of the publication in the snippet above?       | Yes                        |


# Dataset Reconstruction

The original `Synthetic` and `Hybrid` datasets that contain the Snippets with their metadata can be found in the `./data/raa_synthetic_dataset.json` and `data\raa_hybrid_dataset.json` files.

To **augment the datasets**, reconstruct the `Synthetic` and `Hybrid` **transformed QA pairs**, and split them into **train, dev and test sets**, please follow these steps:

1. Run `augment_split_training_data_synthetic.py` to augment and transform to the instruction-based QA task the Synthetic Dataset and create the train, dev and test sets.
    - This will also download the paraphrase model if not in the cache folder, using the `paraphrase_model` script.

2. Run `augment_split_training_data_hybrid_diff.py` to augment and transform to the instruction-based QA task only the real-paper instances from the Hybrid Dataset.

3. Run `combine_hybrid_diff.py` to combine the synthetic and real data to create the final augmented and transformed Hybrid Dataset train, dev and test sets.

The final train, dev and test set files for the Synthetic and Hybrid datasets are:

| Dataset | File |
| --- | --- |
| **Synthetic** | `raa_synthetic_dataset_aug_transformed_train.json` |
|  | `raa_synthetic_dataset_aug_transformed_dev.json` |
|  | `raa_synthetic_dataset_aug_transformed_test.json` |
| **Hybrid** | `raa_hybrid_dataset_aug_transformed_train.json` |
|  | `raa_hybrid_dataset_aug_transformed_dev.json` |
|  | `raa_hybrid_dataset_aug_transformed_test.json` |

# Dataset Statistics

We provide three scripts `create_dataset_statistics_synthetic`, `create_dataset_statistics_hybrid` and `print_latex_dataset_statistics` which construct the statistics for the `Synthetic` and `Hybrid` datasets and convert them to LATEX format.

Here we present them in markdown:

### SYNTHETIC DATASET - Snippets
|                                   |                 |                  |             | **Original** |                  |             |                 |                  |               |                 |                  |             | **Augmented** |                  |             |                 |                  |              |
|------------------------------------|------------------|-------------------|--------------|-------------------|-------------------|--------------|------------------|-------------------|----------------|------------------|-------------------|--------------|--------------------|-------------------|--------------|------------------|-------------------|--------------|
|                                   | **Train**   |                  |             | **Dev**      |                  |             | **Test**    |                  |               | **Train**   |                  |             | **Dev**       |                  |             | **Test**    |                  |              |
|                                   | **dataset** | **software** | **all** | **dataset**  | **software** | **all** | **dataset** | **software** | **all**  | **dataset** | **software** | **all** | **dataset**   | **software** | **all** | **dataset** | **software** | **all** |
| **Number of instances**       | 554              | 647               | 1201         | 98                | 123               | 221          | 89               | 105               | 194            | 1981             | 2247              | 4228         | 292                | 335               | 627          | 282              | 309               | 591          |
| **Number valid**              | 476              | 584               | 1060         | 87                | 107               | 194          | 69               | 98                | 167            | 1694             | 2022              | 3716         | 258                | 287               | 545          | 211              | 295               | 506          |
| **Number w. name**            | 401              | 468               | 869          | 78                | 90                | 168          | 58               | 82                | 140            | 1422             | 1614              | 3036         | 226                | 237               | 463          | 171              | 243               | 414          |
| **Number w. version**         | 42               | 235               | 277          | 11                | 61                | 72           | 0                | 57                | 57             | 122              | 762               | 884          | 33                 | 151               | 184          | 0                | 178               | 178          |
| **Number w. license**         | 142              | 192               | 334          | 38                | 46                | 84           | 20               | 47                | 67             | 519              | 616               | 1135         | 119                | 128               | 247          | 79               | 139               | 218          |
| **Number w. URL**             | 224              | 171               | 395          | 38                | 38                | 76           | 16               | 20                | 36             | 764              | 593               | 1357         | 95                 | 60                | 155          | 28               | 48                | 76           |
| **Number w. ownership**       | 158              | 142               | 300          | 35                | 10                | 45           | 29               | 28                | 57             | 586              | 499               | 1085         | 118                | 30                | 148          | 115              | 81                | 196          |
| **Number w. usage**           | 296              | 469               | 765          | 57                | 88                | 145          | 38               | 74                | 112            | 1016             | 1631              | 2647         | 160                | 222               | 382          | 88               | 241               | 329          |
| **Number of unique snippets** | 148              | 176               | 240          | 25                | 25                | 32           | 25               | 24                | 33             | 1589             | 1796              | 3298         | 232                | 258               | 474          | 230              | 247               | 463          |


### SYNTHETIC DATASET - QA Pairs
|                                    |                 |                  |             | **Original** |                  |             |                 |                  |               |                 |                  |             | **Augmented** |                  |             |                 |                  |              |
|-------------------------------------|------------------|-------------------|--------------|-------------------|-------------------|--------------|------------------|-------------------|----------------|------------------|-------------------|--------------|--------------------|-------------------|--------------|------------------|-------------------|--------------|
|                                    | **Train**   |                  |             | **Dev**      |                  |             | **Test**    |                  |               | **Train**   |                  |             | **Dev**       |                  |             | **Test**    |                  |              |
|                                    | **dataset** | **software** | **all** | **dataset**  | **software** | **all** | **dataset** | **software** | **all**  | **dataset** | **software** | **all** | **dataset**   | **software** | **all** | **dataset** | **software** | **all** |
| **Number of instances**        | 3419             | 4193              | 7612         | 620               | 765               | 1385         | 509              | 706               | 1215           | 12147            | 14432             | 27639        | 1840               | 2057              | 4021         | 1572             | 2103              | 3815         |
| **Number of snippets w. tags** | 554              | 647               | 1201         | 98                | 123               | 221          | 89               | 105               | 194            | 1981             | 2247              | 4228         | 292                | 335               | 627          | 282              | 309               | 591          |
| **Number valid**               | 476              | 584               | 1060         | 87                | 107               | 194          | 69               | 98                | 167            | 1694             | 2022              | 3716         | 258                | 287               | 545          | 211              | 295               | 506          |
| **Number w. name**             | 401              | 468               | 869          | 78                | 90                | 168          | 58               | 82                | 140            | 1422             | 1614              | 3036         | 226                | 237               | 463          | 171              | 243               | 414          |
| **Number w. version**          | 42               | 235               | 277          | 11                | 61                | 72           | 0                | 57                | 57             | 122              | 762               | 884          | 33                 | 151               | 184          | 0                | 178               | 178          |
| **Number w. license**          | 142              | 192               | 334          | 38                | 46                | 84           | 20               | 47                | 67             | 519              | 616               | 1135         | 119                | 128               | 247          | 79               | 139               | 218          |
| **Number w. URL**              | 224              | 171               | 395          | 38                | 38                | 76           | 16               | 20                | 36             | 764              | 593               | 1357         | 95                 | 60                | 155          | 28               | 48                | 76           |
| **Number w. ownership**        | 158              | 142               | 300          | 35                | 10                | 45           | 29               | 28                | 57             | 586              | 499               | 1085         | 118                | 30                | 148          | 115              | 81                | 196          |
| **Number w. usage**            | 296              | 469               | 765          | 57                | 88                | 145          | 38               | 74                | 112            | 1016             | 1631              | 2647         | 160                | 222               | 382          | 88               | 241               | 329          |
| **Number of unique snippets**  | 554              | 647               | 240          | 98                | 123               | 32           | 89               | 105               | 33             | 1981             | 2247              | 3298         | 292                | 335               | 474          | 282              | 309               | 463          |
| **Number of special-type**     | 0                | 0                 | 0            | 0                 | 0                 | 0            | 0                | 0                 | 0              | 489              | 616               | 1059         | 64                 | 71                | 124          | 64               | 84                | 140          |


### HYBRID DATASET - Snippets
|                                   |                 |                  |             | **Original** |                  |             |                 |                  |               |                 |                  |             | **Augmented** |                  |             |                 |                  |              |
|------------------------------------|------------------|-------------------|--------------|-------------------|-------------------|--------------|------------------|-------------------|----------------|------------------|-------------------|--------------|--------------------|-------------------|--------------|------------------|-------------------|--------------|
|                                   | **Train**   |                  |             | **Dev**      |                  |             | **Test**    |                  |               | **Train**   |                  |             | **Dev**       |                  |             | **Test**    |                  |              |
|                                   | **dataset** | **software** | **all** | **dataset**  | **software** | **all** | **dataset** | **software** | **all**  | **dataset** | **software** | **all** | **dataset**   | **software** | **all** | **dataset** | **software** | **all** |
| **Number of instances**       | 757              | 1126              | 1883         | 128               | 222               | 350          | 125              | 181               | 306            | 2332             | 3125              | 5457         | 331                | 507               | 838          | 354              | 463               | 817          |
| **Number valid**              | 615              | 951               | 1566         | 108               | 189               | 297          | 93               | 149               | 242            | 1958             | 2712              | 4670         | 286                | 439               | 725          | 258              | 403               | 661          |
| **Number w. name**            | 488              | 769               | 1257         | 88                | 152               | 240          | 75               | 120               | 195            | 1592             | 2199              | 3791         | 238                | 352               | 590          | 194              | 329               | 523          |
| **Number w. version**         | 42               | 235               | 277          | 11                | 61                | 72           | 0                | 57                | 57             | 122              | 762               | 884          | 33                 | 151               | 184          | 0                | 178               | 178          |
| **Number w. license**         | 142              | 201               | 343          | 38                | 55                | 93           | 20               | 47                | 67             | 519              | 633               | 1152         | 119                | 131               | 250          | 79               | 139               | 218          |
| **Number w. URL**             | 225              | 173               | 398          | 38                | 38                | 76           | 16               | 24                | 40             | 767              | 601               | 1368         | 95                 | 60                | 155          | 28               | 63                | 91           |
| **Number w. ownership**       | 175              | 235               | 410          | 36                | 39                | 75           | 33               | 53                | 86             | 620              | 673               | 1293         | 119                | 75                | 194          | 131              | 138               | 269          |
| **Number w. usage**           | 427              | 770               | 1197         | 77                | 158               | 235          | 60               | 115               | 175            | 1262             | 2208              | 3470         | 186                | 344               | 530          | 130              | 332               | 462          |
| **Number of unique snippets** | 194              | 230               | 298          | 32                | 34                | 41           | 32               | 34                | 43             | 1815             | 2337              | 4027         | 257                | 369               | 605          | 278              | 341               | 598          |


### HYBRID DATASET - QA Pairs
|                                    |                 |                  |             | **Original** |                  |             |                 |                  |               |                 |                  |             | **Augmented** |                  |             |                 |                  |              |
|-------------------------------------|------------------|-------------------|--------------|-------------------|-------------------|--------------|------------------|-------------------|----------------|------------------|-------------------|--------------|--------------------|-------------------|--------------|------------------|-------------------|--------------|
|                                    | **Train**   |                  |             | **Dev**      |                  |             | **Test**    |                  |               | **Train**   |                  |             | **Dev**       |                  |             | **Test**    |                  |              |
|                                    | **dataset** | **software** | **all** | **dataset**  | **software** | **all** | **dataset** | **software** | **all**  | **dataset** | **software** | **all** | **dataset**   | **software** | **all** | **dataset** | **software** | **all** |
| **Number of instances**        | 4456             | 6882              | 11338        | 776               | 1356              | 2132         | 689              | 1088              | 1777           | 14082            | 19458             | 34808        | 2047               | 3141              | 5335         | 1926             | 2905              | 4993         |
| **Number of snippets w. tags** | 757              | 1126              | 1883         | 128               | 222               | 350          | 125              | 181               | 306            | 2332             | 3125              | 5457         | 331                | 507               | 838          | 354              | 463               | 817          |
| **Number valid**               | 615              | 951               | 1566         | 108               | 189               | 297          | 93               | 149               | 242            | 1958             | 2712              | 4670         | 286                | 439               | 725          | 258              | 403               | 661          |
| **Number w. name**             | 488              | 769               | 1257         | 88                | 152               | 240          | 75               | 120               | 195            | 1592             | 2199              | 3791         | 238                | 352               | 590          | 194              | 329               | 523          |
| **Number w. version**          | 42               | 235               | 277          | 11                | 61                | 72           | 0                | 57                | 57             | 122              | 762               | 884          | 33                 | 151               | 184          | 0                | 178               | 178          |
| **Number w. license**          | 142              | 201               | 343          | 38                | 55                | 93           | 20               | 47                | 67             | 519              | 633               | 1152         | 119                | 131               | 250          | 79               | 139               | 218          |
| **Number w. URL**              | 225              | 173               | 398          | 38                | 38                | 76           | 16               | 24                | 40             | 767              | 601               | 1368         | 95                 | 60                | 155          | 28               | 63                | 91           |
| **Number w. ownership**        | 175              | 235               | 410          | 36                | 39                | 75           | 33               | 53                | 86             | 620              | 673               | 1293         | 119                | 75                | 194          | 131              | 138               | 269          |
| **Number w. usage**            | 427              | 770               | 1197         | 77                | 158               | 235          | 60               | 115               | 175            | 1262             | 2208              | 3470         | 186                | 344               | 530          | 130              | 332               | 462          |
| **Number of unique snippets**  | 757              | 1126              | 298          | 128               | 222               | 41           | 125              | 181               | 43             | 2332             | 3125              | 4027         | 331                | 507               | 605          | 354              | 463               | 598          |
| **Number of special-type**     | 0                | 0                 | 0            | 0                 | 0                 | 0            | 0                | 0                 | 0              | 575              | 773               | 1267         | 73                 | 90                | 147          | 72               | 106               | 162          |

# LoRA Training

For the training of the LoRAs on the `Flan-T5 Base` models we have used the **Parameter-Efficient Fine-Tuning (PEFT)** and **Accelerate libraries**, along with **Weights & Biases (wandb) Sweep Configuration**.

To run the LoRA training on the `Synthetic` and `Hybrid datasets`, run the files `flan_t5_lora_accelerate_sweep_synthetic.py` and `flan_t5_lora_accelerate_sweep_hybrid.py` respectively, and use the YAML files in the `./yaml_files` folder to configure the training parameters (dataset paths, output path, trial numbers, wandb project name, etc).

The models are loaded using the `transformers` library from [Huggingface](https://huggingface.co/docs/transformers/index).

When the training is initialized the `Flan-T5 Base` model will be downloaded in the `./models_cache` folder and the default output for the LoRA weight files are in the `./model_checkpoints` folder.

In this repository we have included the weights of the best runs for the `Synthetic` and `Hybrid datasets`, which we mention in the paper as `LoRA-Sy` and `LoRA-Hy` accordingly:

| Model | Path |
| --- | --- |
| Lora-Sy | `./model_checkpoints/sweep_synthetic/flan_t5_base_lora_LORA_SEQ_2_SEQ_LM_fanciful-sweep-8_16_16_0.4_3` |
| Lora-Hy | `./model_checkpoints/sweep_hybrid/flan_t5_base_lora_LORA_SEQ_2_SEQ_LM_smooth-sweep-1_16_16_0.4_4` |

The hyper-parameter tuning configuration for the `LoRA-Sy` Sweep is defined inside the `flan_t5_lora_accelerate_sweep_synthetic.py` file:

```python
sweep_config = {
    'method': 'bayes'
}
metric = {
    'name': 'best_eval_loss',
    'goal': 'minimize'   
}
sweep_config['metric'] = metric
parameters_dict = {
    'optimizer': {
        'value': 'adamw'
        },
    'r': {
        'values': [4, 8, 16, 32]
    },
    'lora_alpha': {
        'values': [8, 16, 32, 64]
    },
    'lora_dropout': {
        'values': [0.1, 0.2, 0.25, 0.3, 0.35, 0.4]
    },
    'learning_rate': {
        'distribution': 'uniform',
        'min': 0.000005, 
        'max': 0.0005
    },
    'epochs': {
        'values': [5, 10, 15]
    },
    'batch_size': {
        'value': yaml_file['batch_size']['value']
    },
    'max_length': {
        'value': 1024
    },
    'model_name': {
        'value': yaml_file['model_name']['value']
    }
}
```

As mentioned in the paper, the `LoRA-Hy` model, uses the best hyper-parameters from the `LoRA-Sy` sweep:

```python
sweep_config = {
    'method': 'bayes'
}
metric = {
    'name': 'best_eval_loss',
    'goal': 'minimize'   
}
sweep_config['metric'] = metric
parameters_dict = {
    'optimizer': {
        'value': 'adamw'
        },
    'r': {
        'values': [16]
    },
    'lora_alpha': {
        'values': [16]
    },
    'lora_dropout': {
        'values': [0.4]
    },
    'learning_rate': {
        'values': [0.00008187494780772831]
    },
    'epochs': {
        'values': [5]
    },
    'batch_size': {
        'value': yaml_file['batch_size']['value']
    },
    'max_length': {
        'value': 1024
    },
    'model_name': {
        'value': yaml_file['model_name']['value']
    }
}
```

# Model APIs

In order to infer the trained models and compare them to the base `Flan-T5 models (Base, XL)`, we have created a a `Web API` and `REST API` for each model:

| Model | API Script | Model Loader & Inference Script |
| --- | --- | --- |
| Flan-T5 Base | `web_api_base.py` | `flan_t5_base_api.py` |
| Flan-T5 XL | `web_api_xl.py` | `flan_t5_xl_api.py` |
| LoRA-Sy | `web_api_lora_sy.py` | `flan_t5_base_lora_api_synthetic.py` |
| LoRA-Hy | `web_api_lora_hy.py` | `flan_t5_base_lora_api_hybrid.py` |

The `Web API` uses the `Flask` and `Waitress` libraries to serve a simple web interface to test the model (accessible by `HOST:PORT/web_api`), which can be customized in the `./templates` folder. 

It also supports REST API requests in the `HOST:PORT/infer_sentence` endpoint, where the prompt (`text`) and the generation parameters (`gen_config`) can be specified. For further information about the generation parameters you can read the Huggingface [documentation](https://huggingface.co/docs/transformers/main_classes/text_generation).

# Evaluation on test set

In the `evaluate_on_test_set.py` script, we evaluate the test set for each combination of model & dataset.

To recreate the results you can comment and uncomment each model and dataset to create the evaluation results files that are stored by default in the `./evaluation_results` folder.

Results on the `Synthetic` dataset:

|                     |                         |                     |             |                         |                     |                |                         |                     |                |                         |                     |                |
|---------------------|-------------------------|---------------------|-------------|-------------------------|---------------------|----------------|-------------------------|---------------------|----------------|-------------------------|---------------------|----------------|
|                    | **Flan T5 base**   |                    |            | **Flan T5 XL**     |                    |               | **LoRA-Sy**        |                    |               | **LoRA-Hy**        |                    |               |
|                    | **Identification** | **Extraction** |            | **Identification** | **Extraction** |               | **Identification** | **Extraction** |               | **Identification** | **Extraction** |               |
|                    | **F1**             | **EM**         | **LM** | **F1**             | **EM**         | **LM**    | **F1**             | **EM**         | **LM**    | **F1**             | **EM**         | **LM**    |
| **Valid**      | 0.841                   | -                   | -           | 0.870                   | -                   | -              | 0.967                   | -                   | -              | **0.974**          | -                   | -              |
| **Name**       | 0.358                   | 0.709               | 0.835       | 0.681                   | 0.787               | 0.900          | **0.887**          | **0.917**      | **0.962** | 0.876                   | 0.905               | 0.952          |
| **License**    | 0.926                   | 0.502               | 0.813       | 0.928                   | 0.635               | 0.778          | **0.946**          | **0.700**      | **0.818** | 0.944                   | 0.685               | **0.818** |
| **Version**    | 0.677                   | 0.620               | 0.816       | 0.942                   | 0.687               | **0.865** | 0.975                   | 0.620               | 0.626          | **0.979**          | **0.755**      | 0.767          |
| **URL**        | 0.677                   | 0.342               | 0.355       | 0.980                   | 0.539               | 0.566          | 0.981                   | 0.618               | 0.645          | **0.982**          | **0.632**      | **0.658** |
| **Usage**      | 0.377                   | -                   | -           | 0.772                   | -                   | -              | 0.911                   | -                   | -              | **0.914**          | -                   | -              |
| **Provenance** | 0.537                   | -                   | -           | 0.647                   | -                   | -              | 0.939                   | -                   | -              | **0.961**          | -                   | -              |

Results on the `Hybrid` dataset:

|                     |                         |                     |             |                         |                     |                |                         |                     |                |                         |                     |                |
|---------------------|-------------------------|---------------------|-------------|-------------------------|---------------------|----------------|-------------------------|---------------------|----------------|-------------------------|---------------------|----------------|
|                    | **Flan T5 base**   |                    |            | **Flan T5 XL**     |                    |               | **LoRA-Sy**        |                    |               | **LoRA-Hy**        |                    |               |
|                    | **Identification** | **Extraction** |            | **Identification** | **Extraction** |               | **Identification** | **Extraction** |               | **Identification** | **Extraction** |               |
|                    | **F1**             | **EM**         | **LM** | **F1**             | **EM**         | **LM**    | **F1**             | **EM**         | **LM**    | **F1**             | **EM**         | **LM**    |
| **Valid**      | 0.766                   | -                   | -           | 0.822                   | -                   | -              | 0.938                   | -                   | -              | **0.960**          | -                   | -              |
| **Name**       | 0.375                   | 0.613               | 0.771       | 0.602                   | 0.698               | 0.830          | 0.832                   | 0.820               | 0.907          | **0.852**          | **0.840**      | **0.911** |
| **License**    | 0.948                   | 0.502               | 0.813       | 0.953                   | 0.635               | 0.778          | **0.963**          | **0.700**      | **0.818** | 0.962                   | 0.685               | **0.818** |
| **Version**    | 0.738                   | 0.620               | 0.816       | 0.935                   | 0.687               | **0.865** | 0.973                   | 0.538               | 0.571          | **0.983**          | **0.755**      | 0.767          |
| **URL**        | 0.723                   | 0.330               | 0.352       | 0.968                   | 0.495               | 0.527          | 0.973                   | 0.538               | 0.571          | **0.982**          | **0.571**      | **0.604** |
| **Usage**      | 0.286                   | -                   | -           | 0.765                   | -                   | -              | 0.898                   | -                   | -              | **0.921**          | -                   | -              |
| **Provenance** | 0.523                   | -                   | -           | 0.650                   | -                   | -              | 0.895                   | -                   | -              | **0.926**          | -                   | -              |


We have also created a `compare_test_results.py` scripts that loads the results of all models in the test set of the `Synthetic` and `Hybrid` datasets and creates a consolidated JSON file for each instance in the datasets, so that we have easily perform Qualitative Analysis on the results.

Example:
```json
 {
  "id": 446,
  "snippet": "To train HeadlineSense, our <m>news headline classification model</m>, we used the News Headlines Dataset, which consists of headlines from news articles. The dataset is widely used for text classification tasks. It is released under the Open Data Commons Attribution License (ODC-BY).",
  "question": "Is the software defined in the <m> and </m> tags introduced or created by the authors of the publication in the snippet above?",
  "answer": "Yes",
  "base_prediction": "No",
  "xl_prediction": "No",
  "synthetic_prediction": "Yes",
  "hybrid_prediction": "Yes"
 },
```