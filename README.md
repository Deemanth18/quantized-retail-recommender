# Quantized Retail Recommender

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/gzbaw6hby2/badge.svg)](https://ask.thili.ai/b/gzbaw6hby2)

This project develops a smart retail recommender that identifies frequently bought items. It leverages a small language model, trained on synthetic shopping basket data, to predict next items. To prepare the model for efficient deployment on edge devices or faster loading, its footprint is reduced using quantization techniques.

## What you are building

**Goal.** You will build a smart retail recommender that identifies frequently bought items. The project culminates in a smaller, functional recommendation model achieved by training a language model on shopping baskets and then applying post-training quantization to its weights.

**Done when.** You will have a functional recommendation model that uses the technique from `lfretail-lfmarketbasket` to train a language model on shopping baskets for next-item prediction. You will then apply the technique from `lfvision-lfquantization` to apply post-training quantization to the model's weights, demonstrating size reduction and evaluating its accuracy impact.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare synthetic shopping basket data for training**  
      Hint: Prepare data for language model training.
- [x] **2. Continue pretraining a small LM on the basket data**  
      Hint: Continue pretraining the language model.
- [x] **3. Implement next-item prediction and itemset probing**  
      Hint: Use the language model for next-item prediction and itemset probing.
- [x] **4. Measure the baseline size and accuracy of the model**  
      Hint: Measure initial model size and accuracy.
- [x] **5. Apply dynamic and static post-training quantization to the model**  
      Hint: Apply post-training dynamic and static quantization.
- [ ] **6. Demonstrate size reduction and evaluate accuracy impact**  
      Hint: Demonstrate size reduction and evaluate accuracy changes.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Analyze the trade-off between model size, inference speed, and accuracy more rigorously across different quantization settings.
- Investigate the impact of quantization on the quality of specific item recommendations.
- Experiment with different bit-widths for quantization if the drills support it, and observe the effects.
