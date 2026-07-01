# RIHA: Report-Image Hierarchical Alignment for Radiology Report Generation

Official implementation of **"RIHA: Report-Image Hierarchical Alignment for Radiology Report Generation"** (IEEE JBHI, 2026).

RIHA is an end-to-end framework that performs **multi-level alignment** between chest X-ray images and their radiology reports across **paragraph, sentence, and word** levels. It introduces:

- **Visual Feature Pyramid (VFP)** — multi-scale visual features from shallow/middle/high layers of a ResNet-101 backbone.
- **Text Feature Pyramid (TFP)** — multi-granularity text features (paragraph / sentence / word) encoded with a frozen Bio_ClinicalBERT.
- **Cross-modal Hierarchical Alignment (CHA)** — aligns visual and textual features at each granularity via **optimal transport** (Wasserstein distance). Used only during training; removable at inference.
- **Relative Positional Encoding (RPE)** — added to the decoder to strengthen token-level alignment.

> This codebase is built on top of [R2Gen (Chen et al., EMNLP 2020)](https://github.com/cuhksz-nlp/R2Gen); the memory-driven Transformer backbone and the `pycocoevalcap` evaluation package originate from that project.

## Requirements

```bash
conda create -n riha python=3.9
conda activate riha
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

A working **Java runtime** is required by `pycocoevalcap` (METEOR and the Stanford tokenizer jars).

## Datasets

We use two public chest X-ray datasets:

- **IU X-Ray** — download and place under `data/iu_xray/` (`images/` + `annotation.json`).
- **MIMIC-CXR** — download and place under `data/mimic_cxr/` (`images/` + `mimic-annotation.json`).

Both follow the R2Gen data split. Only the *findings* section is used as the target text.

## Pipeline

### 1. Prepare multi-granularity annotations

Split each report into sentences (`report_sentences`) and extract key phrases (`report_words`).
Sentence splitting relies on NLTK; phrase extraction is done by:

```bash
python extract_phrase.py \
    --input data/iu_xray/annotation_sentence.json \
    --output data/iu_xray/annotation_phrase.json
```

### 2. Pre-extract text features (TFP)

Encode paragraph / sentence / word segments with Bio_ClinicalBERT and cache them as JSON:

```bash
python -m modules.text_extractor \
    --report_path   data/iu_xray/annotation.json \
    --sentence_path data/iu_xray/annotation_sentence.json \
    --word_path     data/iu_xray/annotation_phrase.json \
    --output_dir    data/features
```

This produces `data/features/{paragraph,sentence,word}_features.json`, consumed by the CHA alignment loss.

### 3. Train

```bash
bash scripts/run_iu_xray.sh      # IU X-Ray
bash scripts/run_mimic_cxr.sh    # MIMIC-CXR
```

### 4. Evaluate

```bash
bash scripts/test_iu_xray.sh     # set --resume to your checkpoint
```

## Key arguments

| Argument | Description |
| --- | --- |
| `--use_alignment_loss` | Enable the CHA hierarchical alignment loss (RIHA). |
| `--alignment_weight` | Overall weight of the alignment loss. |
| `--high/mid/low_level_weight` | Paragraph / sentence / word alignment weights (α, γ, β). |
| `--ot_impl` | Optimal-transport solver (`pot-uot-l2`, `pot-sinkhorn-l2`). |
| `--ot_reg`, `--ot_tau` | OT regularization and marginal-penalization coefficients. |

## Repository layout

```
main.py                 # training entry point
test.py                 # evaluation entry point
extract_phrase.py       # phrase extraction for word-level TFP
models/r2gen.py         # RIHA model (VFP + encoder-decoder wiring)
modules/
  multilevel_visual_extractor.py   # Visual Feature Pyramid (VFP)
  text_extractor.py                # Bio_ClinicalBERT Text Feature Pyramid (TFP)
  alignment.py                     # Cross-modal Hierarchical Alignment (CHA, optimal transport)
  encoder_decoder.py               # Transformer decoder with Relative Positional Encoding (RPE)
  loss.py, trainer.py, ...         # losses, training loop, data, metrics
pycocoevalcap/          # NLG metric implementations (from R2Gen)
scripts/                # train / test shell scripts
```

## Citation

```bibtex
@article{chen2026riha,
  title   = {RIHA: Report-Image Hierarchical Alignment for Radiology Report Generation},
  author  = {Chen, Yucheng and Yu, Yang and Shi, Yufei and Xiong, Conghao and Yang, Xulei and Yeo, Si Yong},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  doi     = {10.1109/JBHI.2026.3670023}
}
```

## Acknowledgements

This work builds on [R2Gen](https://github.com/cuhksz-nlp/R2Gen). We thank the authors for releasing their code.
