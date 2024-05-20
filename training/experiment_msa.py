from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk
import argparse
from whisper_utils import DataCollatorSpeechSeq2SeqWithPadding, compute_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_size', required=True, help="Train size between 0 and 1")
    args = parser.parse_args()
    train_size = float(args.train_size)
    common_voice = load_from_disk('common_voice_arabic_preprocessed/')
    common_voice["train"] = common_voice["train"].train_test_split(test_size=0.2, seed=42)

    train_indices = range(int(len(common_voice["train"]["train"]) * train_size))
    common_voice["train"]["train"] = common_voice["train"]["train"].shuffle(seed=42).select(train_indices)
    common_voice["test"] = common_voice["train"]["test"]
    common_voice["train"] = common_voice["train"]["train"]

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="Arabic", task="transcribe"
    )
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
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()