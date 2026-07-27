# TelecomLM: A Domain-Specific Decoder-Only Transformer for Telecom Question Answering

This work is based on a decoder-only Transformer language model, TelecomLM, built entirely from scratch for the telecommunications and electronics domain. The first stage pre-trains the model from scratch on a large corpus of telecom- and electronics-related text collected from Wikipedia and arXiv, using nothing more than a standard next-token-prediction objective. The second stage fine-tunes this pre-trained model on a question-answering dataset built specifically around the undergraduate Electronics and Telecommunication Engineering (ETE) curriculum, so that the model can answer curriculum-style questions directly rather than simply continuing a passage of text.

Two configurations of TelecomLM were trained under identical conditions to study how model size interacts with a fixed, modestly sized domain corpus: a 49.9-million-parameter model (`d_model=512, heads=8, decoder blocks=6`) and a 137.5-million-parameter model (`d_model=1024, heads=16, decoder blocks=6`). All training was carried out entirely on free-tier Kaggle GPU sessions, with checkpointing and history logs designed to let training pause and resume cleanly across multiple sessions.

The code is written in Python using **PyTorch**, with tokenization handled by the Hugging Face **tokenizers** library. The pre-training corpus and the ETE curriculum question-answering dataset used for this work are available at:
`<add your Kaggle dataset link(s) here>`

## Repository Contents

| File | Description |
|---|---|
| `telecomlm49m_pretrain.py` | Builds the decoder-only Transformer from scratch, trains a Byte-Pair-Encoding tokenizer, packs the corpus into contiguous token windows, and pre-trains the model with a next-token-prediction objective across multiple resumable Kaggle sessions. |
| `telecom_llm_finetune_49m.py` | Loads the pre-trained checkpoint and fine-tunes it on the ETE curriculum question-answering dataset, using early stopping and a one-cycle learning-rate schedule. |
| `requirements.txt` | Python packages required to run both scripts. |

Both scripts were originally developed and run as Kaggle notebooks; they are provided here as plain `.py` files, with `# %%` markers separating each original notebook cell. These markers are recognised by VS Code, PyCharm, and Jupyter as individual runnable cells, so the code can still be run interactively section by section, or as a normal top-to-bottom script.

## Setup

```bash
git clone https://github.com/antorhajra/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
python telecomlm49m_pretrain.py
python telecom_llm_finetune_49m.py
```

Both scripts were originally written and trained on Kaggle notebooks with free-tier GPU sessions, so file paths inside them (for example, `/kaggle/input/...` and `/kaggle/working/...`) will need to be updated to match wherever you place your own dataset and output folders if you are running them outside Kaggle.

## Method Summary

- **Architecture:** decoder-only Transformer (embedding → learned positional encoding → masked multi-head self-attention → position-wise feed-forward, repeated across stacked decoder blocks → final layer normalization → output projection), following the design introduced by Vaswani et al. and adapted to the generative pre-training paradigm introduced by Radford et al. for GPT-1.
- **Tokenizer:** Byte Pair Encoding, 30,000-token vocabulary, trained once on the pre-training corpus and reused unchanged for fine-tuning.
- **Pre-training data:** telecom- and electronics-domain text gathered from Wikipedia and arXiv.
- **Fine-tuning data:** a question-answering dataset written specifically for this work, spanning the ETE curriculum, in `instruction`/`input`/`output` format.
- **Training:** AdamW optimizer, one-cycle learning-rate schedule, mixed-precision training, gradient accumulation, and session-based checkpointing to work within Kaggle's free-tier GPU quotas.
- **Evaluation:** ROUGE-1/2/L, METEOR, BERTScore, and perplexity, together with direct reading of generated answers, compared against fine-tuned general-purpose baselines (Pythia-70M, GPT-2, DistilGPT2) and an external telecom question-answering benchmark.

## Author

Antor Hajra

This work was carried out as part of a final-year undergraduate thesis in Electronics and Telecommunication Engineering (ETE), Rajshahi University of Engineering & Technology (RUET).
