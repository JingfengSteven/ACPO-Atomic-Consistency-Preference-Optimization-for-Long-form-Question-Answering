# FactScore Integration
This folder contains our custom utilization of the [FActScore](https://github.com/sewonmin/factscore) evaluation framework for atomic-level factual precision scoring, used in our long-form question answering experiments.

## Run Command

To evaluate model generations using FActScore with ChatGPT as the verifier, use the following command:

```
python -m factscore.factscorer \
  --input_path "path to the jsonl file with question-answering pairs" \
  --model_name retrieval+ChatGPT \
  --openai_key "path to the openai_key .txt file" \
