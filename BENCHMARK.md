# Benchmark Analysis

> Detailed evaluation on `datasets/augmented/test.csv` (15,005 samples with simulated STT errors)
> Generated from [`notebooks/full_evaluation.ipynb`](../notebooks/full_evaluation.ipynb)

---

## Table of Contents

1. [Pre-Processing Impact](#1-pre-processing-impact)
2. [Component Comparison](#2-component-comparison)
3. [Post-Processing Impact](#3-post-processing-impact)
4. [Performance by Language](#4-performance-by-language)
5. [Language Detection Pipeline Value](#5-language-detection-pipeline-value)
6. [End-to-End Pipeline](#6-end-to-end-pipeline)

---

## 1. Pre-Processing Impact

Evaluates the impact of pre-processing (STT artifact filter + text normalization) on each component.

### Language Detection: Pre-Processing Impact

| Model | Raw | Pre-processed | Delta |
|---|---|---|---|
| Regex | 67.5% | 67.5% | +0.0% |
| Langdetect | 77.9% | 77.9% | +0.0% |

### Intent Classification: Pre-Processing Impact

| Model | Raw | Pre-processed | Delta |
|---|---|---|---|
| Regex | 65.8% | 65.8% | +0.0% |
| CamemBERT | 32.2% | 32.4% | +0.2% |
| SpaCy | 78.7% | 79.1% | +0.4% |

### Entity Extraction: Pre-Processing Impact

| Model | Raw | Pre-processed | Delta |
|---|---|---|---|
| Regex | 33.1% | 33.3% | +0.2% |
| SpaCy | 27.6% | 27.8% | +0.2% |
| CamemBERT | 16.1% | 16.3% | +0.2% |

**Summary**: Pre-processing provides marginal improvements (+0.2% average). SpaCy intent classifier benefits most (+0.4%).

---

## 2. Component Comparison

Compares all models for each pipeline component (using pre-processed data).

### Language Detection Models

| Model | Accuracy | FR Precision | FR Recall | FR F1 | Latency |
|-------|----------|--------------|-----------|-------|---------|
| Regex | 67.5% | 0.89 | 0.72 | 0.80 | 0.0ms |
| Langdetect | 77.9% | 0.96 | 0.80 | 0.87 | 6.1ms |

### Intent Classification Models

| Model | Accuracy | TRIP F1 | NOT_TRIP F1 | UNKNOWN F1 | Latency |
|-------|----------|---------|-------------|------------|---------|
| Regex | 65.8% | 0.78 | 0.34 | 0.14 | 0.0ms |
| SpaCy | 79.1% | 0.87 | 0.51 | 0.00 | 0.8ms |
| CamemBERT | 32.4% | 0.41 | 0.15 | 0.02 | 0.7ms |

### Entity Extraction Models (No Fuzzy)

| Model | Accuracy | Departure F1 | Destination F1 | Latency |
|-------|----------|--------------|----------------|---------|
| Regex | 33.3% | 0.53 | 0.54 | 0.0ms |
| SpaCy | 27.8% | 0.45 | 0.49 | 0.9ms |
| CamemBERT | 16.3% | 0.33 | 0.33 | 0.7ms |

---

## 3. Post-Processing Impact

Evaluates the impact of Fuzzy post-processing on Entity Extraction.

### Entity Extraction: Fuzzy Post-Processing Impact

| Model | No Fuzzy | +Fuzzy | Delta | Latency (no) | Latency (+) |
|---|---|---|---|---|---|
| Regex | 33.3% | 57.1% | **+23.7%** | 0.0ms | 9.8ms |
| SpaCy | 27.8% | 34.2% | +6.4% | 0.9ms | 2.5ms |
| CamemBERT | 16.3% | 31.2% | +15.0% | 0.7ms | 0.8ms |

**Summary**: Fuzzy post-processing provides the largest improvement for Regex (+23.7%), making it the best entity extraction approach overall (57.1% accuracy).

---

## 4. Performance by Language

Segments evaluation results by ground-truth language to understand where models perform best.

> **Note**: English (~966 samples) and Unknown (~673 samples) subsets have lower statistical significance than French (~13,366 samples).

### Intent Classification by Ground-Truth Language

| Model | FR (13366) | EN (966) | UNK (673) |
|---|---|---|---|
| Regex | 65.3% | 70.9% | 69.1% |
| CamemBERT | 26.5% | 23.7% | 25.7% |
| SpaCy | **82.2%** | 57.1% | 50.5% |

**Observation**: SpaCy excels on French (82.2%) but struggles with English (57.1%) and Unknown (50.5%). Regex is more consistent across languages.

### Entity Extraction by Ground-Truth Language (TRIP only, +Fuzzy)

| Model (+Fuzzy) | FR (9990) | EN (666) | UNK (445) |
|---|---|---|---|
| Regex + Fuzzy | **59.2%** | 32.1% | 52.4% |
| SpaCy + Fuzzy | 35.7% | 20.1% | 29.0% |
| CamemBERT + Fuzzy | 31.3% | 28.5% | 33.3% |

**Observation**: Regex + Fuzzy strongly outperforms others on French (59.2%). CamemBERT shows more consistent cross-language performance but lower overall accuracy.

---

## 5. Language Detection Pipeline Value

Evaluates whether adding Language Detection as a filtering step improves downstream performance.

**Comparison:**
- **Baseline**: Run IC/EE on all samples
- **LD-Filtered**: Run LD first, then IC/EE only on LD-predicted-French samples

### Intent Classification: LD Pipeline Value

| Model | Full (15005) | LD-Filtered (11077) | Delta |
|---|---|---|---|
| Regex | 65.8% | 67.2% | +1.4% |
| CamemBERT | 32.4% | 48.8% | **+16.4%** |
| SpaCy | 79.1% | 81.6% | +2.5% |

**Observation**: CamemBERT benefits most from LD filtering (+16.4%), likely because it's trained on French text. SpaCy also improves (+2.5%).

### Entity Extraction: LD Pipeline Value

| Model (+Fuzzy) | Full TRIP (11101) | LD-Filtered TRIP (8740) | Delta |
|---|---|---|---|
| Regex + Fuzzy | 57.1% | **59.8%** | +2.7% |
| SpaCy + Fuzzy | 34.2% | 36.0% | +1.9% |
| CamemBERT + Fuzzy | 31.2% | 28.3% | -2.9% |

**Observation**: LD filtering helps Regex (+2.7%) and SpaCy (+1.9%) but hurts CamemBERT (-2.9%). Recommended: use LD filtering with Regex + Fuzzy pipeline.

---

## 6. End-to-End Pipeline

Summary of the recommended pipeline configuration.

### End-to-End Pipeline Summary

| Pipeline Stage | Best Model | Accuracy | Latency |
|----------------|------------|----------|---------|
| Pre-Processing | STT Filter + Normalizer | - | <1ms |
| Language Detection | Langdetect | 77.9% | 6.1ms |
| Intent Classification | SpaCy | 79.1% | 0.8ms |
| Entity Extraction | Regex + Fuzzy | 57.1% | 9.8ms |

### Pipeline Stage Impact Analysis

| Stage | Average Impact |
|-------|----------------|
| Pre-Processing | +0.2% |
| Fuzzy Post-Processing | **+15.0%** |
| LD Filtering | +3.7% |

**Recommended Configuration:**
- **Pre-Processing**: STT Filter + Normalizer (always enabled)
- **Language Detection**: Langdetect (filter non-French)
- **Intent Classification**: SpaCy (best accuracy on French)
- **Entity Extraction**: Regex + Fuzzy (best overall)

**Total Pipeline Latency**: ~17ms per sample

---

## Methodology

- **Dataset**: `datasets/augmented/test.csv` (15,005 samples with simulated STT errors)
- **Language Distribution**: French 89%, English 6%, Unknown/Other 5%
- **Intent Distribution**: TRIP 74%, NOT_TRIP 25%, UNKNOWN 1%
- **Pre-processing**:
  - STTArtifactFilter: Removes Whisper noise markers (`[music]`, `[inaudible]`, etc.)
  - Preprocessor: Normalizes whitespace, apostrophes, hyphens (preserves case and accents)
- **Metrics**:
  - Accuracy: Overall correct predictions / total samples
  - F1-Score: Harmonic mean of precision and recall (per-class)
  - Latency: Average inference time per sample (ms)
- **Hardware**: NVIDIA GeForce RTX 2060 (GPU), SpaCy on CPU

---

*Last updated: 2026-01-23*
