#!/usr/bin/env python3
import json
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import click

import torch
from peft import prepare_model_for_int8_training, get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, LlamaForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dict = {'easy':'/workspace/model-easy','medium':'/workspace/model-medium', 'hard':'/workspace/model-hard'}



def tokenize_function(tokenizer, example):
    inputs = tokenizer(example,
                     max_length=512,
                     truncation=True,
                     padding=True,
                     return_tensors="pt")
    input_ids = inputs['input_ids']
    return input_ids.to(device)




BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B"

def build_tokenizer(model_id: str = BASE_MODEL_ID):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def build_model(lora_dir, base_id: str = BASE_MODEL_ID):
    base = LlamaForSequenceClassification.from_pretrained(
        base_id,
        num_labels=2,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device)

    # 2. 把 LoRA 权重注入
    model = PeftModel.from_pretrained(base, lora_dir, local_files_only=True)
    model.eval()
    return model

def predict(checkpoint, sentences):
    tokenizer = build_tokenizer()
    input_ids = tokenize_function(tokenizer, sentences)
    lora_model = build_model(checkpoint)
    predict = lora_model(input_ids)
    result = predict.logits.argmax(dim=1).tolist()
    print(result)
    return result


def run_baseline(problems: "pd.DataFrame", output_path: Path, checkpoint: str):
    """
    Write predictions to solution files in the format:
    {
    "changes": [0, 0, 0, 1]
    }

    :param checkpoint:
    :param problems: dictionary of problem files with ids as keys
    :param output_path: output folder to write solution files
    """
    print(f'Write outputs {len(problems)} problems to to {output_path}.')
    for _, i in problems.iterrows():
        output_file = output_path / i["file"].replace("/problem-", "/solution-problem-").replace(".txt", ".json").replace("/train/", "/").replace("/test/", "/").replace("/validation/", "/")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as out:
            paragraphs = i["paragraphs"]
            result = predict(checkpoint, paragraphs)
            prediction = {'changes': [item for item in result]}
            print(f'prediction: {prediction}')
            out.write(json.dumps(prediction))

@click.command()
@click.option('--dataset', default='multi-author-writing-style-analysis-2025/multi-author-writing-spot-check-20250503-training', help='The dataset to run predictions on (can point to a local directory).')
@click.option('--output', default=Path(get_output_directory(str(Path(__file__).parent))), help='The file where predictions should be written to.')
@click.option('--predict', default=0,  help='The prediction to make.')
def main(dataset, output, predict):
    print(f'dataset: {dataset}')
    print(f'output: {output}')
    print(f'predict: {predict}')

    tira = Client()
    # alternatively, you can still load the data programmatically, i.e., without tira.pd.inputs.
    # See https://github.com/pan-webis-de/pan-code/blob/master/clef24/multi-author-analysis/baselines/baseline-static-prediction.py
    input_df = tira.pd.inputs(dataset, formats=["multi-author-writing-style-analysis-problems"])

    for subtask in ["easy", "medium", "hard"]:
        run_baseline(input_df[input_df["task"] == subtask], Path(output), checkpoint_dict[subtask])

if __name__ == '__main__':
    main()