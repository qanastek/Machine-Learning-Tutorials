import os
import argparse
import itertools

import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report

import evaluate
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model path')
parser.add_argument('--task', type=str, help='Task Name')
args = parser.parse_args()

task_dataset = args.task

if task_dataset == "mnli":
    print("Corpus doesn't have any test set!")
    exit(0)

if "mnli" in task_dataset:
    task_dataset = "mnli"

print(transformers.__version__)

dataset = load_dataset("glue", task_dataset)

train_dataset = dataset["train"]

if args.task == "mnli_matched":
    validation_dataset = dataset["validation_matched"]
    test_dataset = dataset["test_matched"]
elif args.task == "mnli_mismatched":
    validation_dataset = dataset["test_mismatched"]
    test_dataset = dataset["test"]
else:
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]

df_train      = train_dataset.to_pandas()
df_validation = validation_dataset.to_pandas()
df_test       = test_dataset.to_pandas()

real_labels = df_train['label'].unique().tolist()
print(real_labels)

f1_metric  = evaluate.load("f1")
acc_metric = evaluate.load("accuracy")

task = "GLUE"
batch_size = 12
EPOCHS = 3

model_checkpoint = str(args.model)

if task_dataset == "stsb":
    problem_type = "regression"
    num_labels = 1
else:
    problem_type = "single_label_classification"
    num_labels = len(real_labels)

print(f"num_labels : {num_labels}")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    problem_type=problem_type,
)

model_name = model_checkpoint.split("/")[-1]

label_list = train_dataset.features["label"].names
print("label_list")
print(label_list)

TRUNCATE   = True
MAX_LENGTH = 512

def preprocess_function(examples):

    if task_dataset == "ax":
        texts = []
        for s1, s2 in zip(examples["premise"], examples["hypothesis"]):
            texts.append(s1 + "[SEP]" + s2)
        return tokenizer(texts, truncation=TRUNCATE, max_length=MAX_LENGTH)

    elif task_dataset == "cola":
        return tokenizer(examples["sentence"], truncation=TRUNCATE, max_length=MAX_LENGTH)

    elif "mnli" in task_dataset:
        texts = []
        for s1, s2 in zip(examples["premise"], examples["hypothesis"]):
            texts.append(s1 + "[SEP]" + s2)
        return tokenizer(texts, truncation=TRUNCATE, max_length=MAX_LENGTH)

    elif task_dataset == "mrpc":
        texts = []
        for s1, s2 in zip(examples["sentence1"], examples["sentence2"]):
            texts.append(s1 + "[SEP]" + s2)
        return tokenizer(texts, truncation=TRUNCATE, max_length=MAX_LENGTH)

    elif task_dataset == "qnli":
        texts = []
        for s1, s2 in zip(examples["question"], examples["sentence"]):
            texts.append(s1 + "[SEP]" + s2)
        return tokenizer(texts, truncation=TRUNCATE, max_length=MAX_LENGTH)

    elif task_dataset == "qqp":
        texts = []
        for s1, s2 in zip(examples["question1"], examples["question2"]):
            texts.append(s1 + "[SEP]" + s2)
        return tokenizer(texts, truncation=TRUNCATE, max_length=MAX_LENGTH)

    elif task_dataset == "rte":
        texts = []
        for s1, s2 in zip(examples["sentence1"], examples["sentence2"]):
            texts.append(s1 + "[SEP]" + s2)
        return tokenizer(texts, truncation=TRUNCATE, max_length=MAX_LENGTH)

    elif task_dataset == "sst2":
        return tokenizer(examples["sentence"], truncation=TRUNCATE, max_length=MAX_LENGTH)

    elif task_dataset == "stsb":
        texts = []
        for s1, s2 in zip(examples["sentence1"], examples["sentence2"]):
            texts.append(s1 + "[SEP]" + s2)
        return tokenizer(texts, truncation=TRUNCATE, max_length=MAX_LENGTH)

    elif task_dataset == "wnli":
        texts = []
        for s1, s2 in zip(examples["sentence1"], examples["sentence2"]):
            texts.append(s1 + "[SEP]" + s2)
        return tokenizer(texts, truncation=TRUNCATE, max_length=MAX_LENGTH)

    else:
        print("Error #133")
        exit(0)

enc_train_dataset      = train_dataset.map(preprocess_function, batched=True)
enc_validation_dataset = validation_dataset.map(preprocess_function, batched=True)
enc_test_dataset       = test_dataset.map(preprocess_function, batched=True)

enc_test_dataset = enc_test_dataset.remove_columns("label").add_column("label", [0 for a in enc_test_dataset["label"]])

print(enc_train_dataset["label"])
print(enc_validation_dataset["label"])
print(enc_test_dataset["label"])

args = TrainingArguments(
    f"GLUE-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    push_to_hub=False,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    load_best_model_at_end=True,
)

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    res_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    res_acc = acc_metric.compute(predictions=predictions, references=labels)

    return {"f1": res_f1["f1"], "accuracy": res_acc["accuracy"]}

trainer = Trainer(
    model,
    args,
    train_dataset=enc_train_dataset,
    eval_dataset=enc_validation_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

trainer.evaluate()

# ------------------ EVALUATION ------------------

predictions, test_real_labels, _ = trainer.predict(enc_test_dataset)
predictions = np.argmax(predictions, axis=1)

LOGS_PATH = f"./outputs"
os.makedirs(LOGS_PATH, exist_ok=True)

log_file = open(f"{LOGS_PATH}/{task_dataset}.tsv","w")

indentifiers = df_test['idx'].unique().tolist()

for i, l in zip(indentifiers, predictions):
    i = str(i)
    l = str(l)
    log_file.write(f"{i}\t{l}\n")

log_file.close()

print(f"Finished!")
