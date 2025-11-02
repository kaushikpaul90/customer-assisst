"""Fine-tune a small QA model on SQuAD subset. Run in Colab or local GPU."""
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='distilbert-base-cased', type=str)
parser.add_argument('--output_dir', default='./models/qa_finetuned', type=str)
parser.add_argument('--num_train_epochs', default=1, type=int)
parser.add_argument('--per_device_train_batch_size', default=8, type=int)
args = parser.parse_args()

dataset = load_dataset('squad', split='train[:1%]')
val = load_dataset('squad', split='validation[:0.5%]')

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

max_length = 384
stride = 128

def prepare_train_features(examples):
    questions = [q.strip() for q in examples['question']]
    inputs = tokenizer(questions, examples['context'], max_length=max_length, truncation='only_second', stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length')
    sample_mapping = inputs.pop('overflow_to_sample_mapping')
    offset_mapping = inputs.pop('offset_mapping')

    start_positions = []
    end_positions = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = inputs.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples['answers'][sample_index]
        if len(answers['answer_start']) == 0:
            start_positions.append(cls_index); end_positions.append(cls_index)
        else:
            start_char = answers['answer_start'][0]
            end_char = start_char + len(answers['text'][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index); end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)
    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs

train_dataset = dataset.map(prepare_train_features, batched=True, remove_columns=dataset.column_names)
val_dataset = val.map(prepare_train_features, batched=True, remove_columns=val.column_names)

training_args = TrainingArguments(output_dir=args.output_dir, evaluation_strategy='epoch',
                                  per_device_train_batch_size=args.per_device_train_batch_size,
                                  per_device_eval_batch_size=8, num_train_epochs=args.num_train_epochs,
                                  save_strategy='epoch', logging_steps=10)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
trainer.train(); trainer.save_model(args.output_dir)
print('Saved to', args.output_dir)
