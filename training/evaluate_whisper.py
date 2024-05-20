from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk
import evaluate

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train_size', required=True)
args = parser.parse_args()
train_size = float(args.train_size)
model_name = f"results_thesis/whisper-small-ar_tsize_{train_size}/checkpoint-5000"

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small", language="Arabic", task="transcribe"
)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="Arabic", task="transcribe"
)

common_voice = load_from_disk('common_voice_arabic_preprocessed/test')


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./whisper-small-ar_tsize_{str(train_size)}",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=448,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)
test_set = common_voice["test"]
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.generation_config.language = "ar"
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    eval_dataset=test_set,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
results = trainer.evaluate(language="ar")
with open(f"results_common_voice{train_size}.json", 'w') as f:
    json.dump(results, f)
