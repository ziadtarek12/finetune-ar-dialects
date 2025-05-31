from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import time
from datasets import load_dataset, Audio

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def benchmark_model(model, processor, dataset):
    total_time = 0
    for sample in dataset:
        audio = sample["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").to(device)
        
        # Ensure input tensor is in the same dtype as the model
        inputs["input_features"] = inputs["input_features"].to(torch_dtype)
        
        start_time = time.time()
        outputs = model.generate(**inputs, language="ar")
        total_time += time.time() - start_time
    return total_time / len(dataset)


# Load Whisper Small
model_small = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-small", torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)
processor_small = AutoProcessor.from_pretrained("openai/whisper-small")

# Load Whisper Large-v3
model_large_v3 = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3", torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)
processor_large_v3 = AutoProcessor.from_pretrained("openai/whisper-large-v3")

common_voice_test_small = load_dataset("mozilla-foundation/common_voice_16_1", "ar", split="test")
common_voice_test_small = common_voice_test_small.cast_column("audio", Audio(sampling_rate=16000))

avg_time_small = benchmark_model(model_small, processor_small, common_voice_test_small)
avg_time_large_v3 = benchmark_model(model_large_v3, processor_large_v3, common_voice_test_small)

print(f"Average inference time for Whisper Small: {avg_time_small:.2f} seconds")
print(f"Average inference time for Whisper Large-v3: {avg_time_large_v3:.2f} seconds")