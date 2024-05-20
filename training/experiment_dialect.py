from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk, concatenate_datasets
import argparse
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from whisper_utils import DataCollatorSpeechSeq2SeqWithPadding, compute_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dialect', required=True, help="all, egyptian, gulf, iraqi, levantine, maghrebi")
    args = parser.parse_args()

    if args.dialect == "all":
        dialect_dataset = load_from_disk('egyptian_train/')
        for d in ["gulf", "iraqi", "levantine", "maghrebi"]:
            train_d = load_from_disk(f'{d}_train/')
            dialect_dataset["train"] = concatenate_datasets([train_d["train"], dialect_dataset["train"]])
            dialect_dataset["test"] = concatenate_datasets([train_d["test"], dialect_dataset["test"]])
    else:
        dialect_dataset = load_from_disk(f'{args.dialect}_train/')

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="Arabic", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "ar"
    model.config.max_length = 512
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./whisper-small-dialect_{args.dialect}",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dialect_dataset["train"],
        eval_dataset=dialect_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()