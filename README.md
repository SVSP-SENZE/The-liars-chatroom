# The Liar's Chatroom - Winning Solution Guide

A pure machine learning solution strictly adhering to all competition rules:
- **No closed/hosted APIs**: Zero calls to OpenAI, Claude, hosted Gemini, or any proprietary external API.
- **Open-weight models only**: Using open-weight reasoning LLMs (such as `Qwen/Qwen2.5-14B-Instruct` or `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`) and discriminative NLI cross-encoders (`microsoft/deberta-v3-large`).
- **Offline / Local Execution**: Fully self-contained pipeline.

---

## 1. Directory Structure

```
The-liars-chatroom/
├── dataset/
│   ├── the-liars-chatroom.zip
│   └── extracted/
│       ├── train.csv (2,500 rows)
│       ├── test.csv (3,500 rows)
│       └── sample_submission.csv
├── data/
│   ├── train_folds.csv        # 5-fold stratified CV split
│   ├── mc_train.json          # 4-choice Multiple Choice training data
│   └── mc_test.json           # 4-choice Multiple Choice test data
├── src/
│   ├── chat_parser.py         # Filters fillers/questions, parses claims
│   ├── make_folds.py          # 5-fold stratified cross-validation
│   ├── prompt_builder.py      # Structured CoT verification prompt engine
│   ├── create_multiple_choice_dataset.py
│   ├── train_deberta.py       # Fine-tunes DeBERTa-v3 Multiple Choice
│   ├── infer_deberta.py       # Generates predictions from DeBERTa-v3
│   ├── kaggle_qwen_reasoner.py# Self-contained open-weight LLM solver
│   ├── ensemble.py            # Majority voting & weighted ensembling
│   └── verify_submission.py   # Strict rule validator for submissions
├── kaggle_liars_chatroom_solution.ipynb  # Ready-to-run Jupyter notebook
└── submissions/
```

---

## 2. Running on Kaggle / Colab (Fastest Path to Submit)

1. Open [Kaggle Notebooks](https://www.kaggle.com/code) or Google Colab.
2. Upload `kaggle_liars_chatroom_solution.ipynb`.
3. In Kaggle, attach the competition dataset (`the-liars-chatroom`) and select **GPU T4 x2** or **P100**.
4. Run all cells:
   - Evaluates validation accuracy on Fold 0.
   - Runs batched 4-bit inference with `Qwen/Qwen2.5-14B-Instruct` across all 3,500 test cases.
   - Automatically validates and outputs `submission.csv`.
5. Download `submission.csv` and submit!

---

## 3. Running Locally / Cloud VM

### A. Format Data & Create Folds
```bash
python src/make_folds.py
python src/create_multiple_choice_dataset.py
```

### B. Train DeBERTa-v3 Multiple Choice
```bash
python src/train_deberta.py --model_name microsoft/deberta-v3-large --val_fold 0 --epochs 3 --batch_size 4
python src/infer_deberta.py --checkpoint_dir checkpoints/deberta_mc/fold_0/best_model --output_csv submissions/sub_deberta.csv
```

### C. Run Open LLM Inference
```bash
python src/run_llm_inference.py --model_name Qwen/Qwen2.5-14B-Instruct --load_in_4bit --output_csv submissions/sub_qwen14b.csv
```

### D. Ensemble & Validate
```bash
python src/ensemble.py --files submissions/sub_qwen14b.csv submissions/sub_deberta.csv --output submissions/final_submission.csv
python src/verify_submission.py submissions/final_submission.csv
```
