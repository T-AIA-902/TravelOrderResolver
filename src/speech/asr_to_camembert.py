import os
import whisper
import torch
from transformers import CamembertTokenizer, CamembertModel

# Chemin vers ton fichier audio
audio_path = "datasets/raw/audio/test_phrase.wav"

if not os.path.exists(audio_path):
    raise FileNotFoundError(f"Le fichier audio {audio_path} n'existe pas !")

# SPEECH TO TEXT (WHISPER)
print("Loading Whisper model...")
whisper_model = whisper.load_model("small")  # small / medium / large

result = whisper_model.transcribe(audio_path, language="fr")
text = result["text"]
print("\n--- Transcription ---")
print(text)

# TEXTE → CAMEMBERT
print("\nLoading CamemBERT...")
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertModel.from_pretrained("camembert-base")

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state
print("\nCamemBERT embeddings shape:")
print(embeddings.shape)
