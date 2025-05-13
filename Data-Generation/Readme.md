# Data Generation
## Run Command
To run data generation, directly use the following command and config
```
python pipline.py
```


### Experiment Directory and Model Path
- **`--base_dir`**: The directory for the entire experiment, where outputs and logs are stored.
- **`--model_id`**: Path to the LLaMA/Phi model used for generation.

### Initial Generation Parameters
- **`--system_prompt_1`**: System prompt for the initial generation phase.
- **`--question_file`**: Path to question files, such as `"train.jsonl"`
- **`--num_answers_per_question`**: Number of initial responses to generate per question, with a default of 30.
- **`--max_new_tokens`**: Maximum number of tokens generated per response.
- **`--temperatures_1`**: Temperature setting to control response randomness.
- **`--clear_cache_1`**: Whether to clear cache after each generation. Set to `True` or `False`.

### Atomic Fact Extraction and Clustering
- **`--atomic_model_path`**: Path to the atomic extractor model, as per the ASC paper (if unchanged).
- **`--cluster_number`**: Number of clusters to use in atomic fact extraction.


### DPO Data Type
- **`--DPO_data_type`**: Select `'P_N'`
  - **`P_N`**: 1 best score, 1 worst score

---
