"""Evaluate a QA model on a small validation subset (EM and F1 via squad metric)"""
import argparse
from datasets import load_dataset, load_metric
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', required=True, type=str)
args = parser.parse_args()

dataset = load_dataset('squad', split='validation[:1%]')
metric = load_metric('squad')
qa_pipe = pipeline('question-answering', model=args.model_dir, tokenizer=args.model_dir)

predictions = []; references = []
for ex in dataset:
    out = qa_pipe({'context': ex['context'], 'question': ex['question']})
    predictions.append({'id': ex['id'], 'prediction_text': out['answer']})
    references.append({'id': ex['id'], 'answers': ex['answers']})

res = metric.compute(predictions=predictions, references=references)
print(res)
