#!/bin/bash
# =============================================================================
# Travel Order Resolver - Download Pre-trained Models
# =============================================================================

set -e

MODELS_DIR="models/production"
mkdir -p "$MODELS_DIR"

echo "Downloading pre-trained models..."

# Download spaCy French transformer model
echo "Downloading spaCy fr_dep_news_trf..."
python -m spacy download fr_dep_news_trf

# Note: Add commands here to download other models as needed
# For example:
# - CamemBERT from HuggingFace
# - Whisper for speech-to-text
# - Custom fine-tuned models

echo ""
echo "To download CamemBERT (for fine-tuning):"
echo "  python -c \"from transformers import CamembertTokenizer, CamembertModel; CamembertTokenizer.from_pretrained('camembert-base'); CamembertModel.from_pretrained('camembert-base')\""
echo ""
echo "To download Whisper (for speech-to-text):"
echo "  python -c \"import whisper; whisper.load_model('base')\""
echo ""

echo "Model download complete!"
