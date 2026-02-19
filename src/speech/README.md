# Speech-to-Text (Whisper)

Module de transcription vocale utilisant OpenAI Whisper en mode offline.

## Installation

```bash
sudo apt-get install -y portaudio19-dev
```
> Library for PortAudio system

```bash
poetry remove openai-whisper && poetry add openai-whisper sounddevice soundfile
```

## Utilisation

### Via le CLI (recommande)

```bash
# Mode interactif avec speech-to-text
python -m src.main --extractor camembert --speech
```

Appuyez sur Entree sans texte pour enregistrer depuis le micro, ou tapez votre demande directement.

> **Note :** Le mode `--speech` necessite **Windows PowerShell**. Le micro n'est pas accessible en WSL.

### Via Python

#### Enregistrement micro (duree fixe)

```python
from src.speech import SpeechTranscriber

transcriber = SpeechTranscriber(model_name="medium")

result = transcriber.transcribe_from_mic(duration=5.0)
print(result.text)      # "Je veux aller de Paris a Lyon"
print(result.language)   # "fr"
```

#### Enregistrement micro (detection de silence)

```python
result = transcriber.transcribe_from_mic_auto(
    silence_threshold=0.01,
    silence_duration=2.0,
    max_duration=15.0,
)
print(result.text)
```

#### Transcription d'un fichier audio

```python
from src.speech import WhisperModel

model = WhisperModel(model_name="medium")
result = model.transcribe("recording.wav")
print(result.text)
```

## Architecture

```
src/speech/
  __init__.py          # Exports publics
  audio_processor.py   # AudioRecorder (enregistrement micro via sounddevice)
  whisper_model.py     # WhisperModel (chargement et inference Whisper)
  transcriber.py       # SpeechTranscriber (orchestrateur haut niveau)
  speech_input.py      # SpeechInput (helper CLI pour main.py)
```

## Modeles Whisper disponibles

| Modele | Parametres | VRAM | Vitesse | Precision |
|--------|-----------|------|---------|-----------|
| tiny   | 39M       | ~1GB | Rapide  | Faible    |
| base   | 74M       | ~1GB | Rapide  | Moyenne   |
| small  | 244M      | ~2GB | Moyen   | Bonne     |
| medium | 769M      | ~5GB | Lent    | Tres bonne|
| large  | 1550M     | ~10GB| Tres lent| Excellente|

Le modele par defaut est `small`. Pour la production, `medium` offre le meilleur rapport qualite/vitesse.

## Limitations

- **WSL** : pas d'acces au micro (PortAudioError device -1). Utiliser Windows PowerShell.
- **CPU** : la transcription est plus lente sans GPU CUDA. Le modele `medium` prend ~5-10s sur CPU.
- **Langue** : le parametre `language="fr"` est utilise par defaut pour optimiser la transcription francaise.
