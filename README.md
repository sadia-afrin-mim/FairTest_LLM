# FairTest_LLM — LLM-Assisted Fairness Testing Framework

An LLM-assisted fairness testing framework that automatically
generates discriminatory test samples for ML classifiers using
zero-shot prompting guided by fairness metrics and SHAP explanations.

---

## Requirements

```bash
pip install openai anthropic scikit-learn shap pandas numpy matplotlib seaborn
```

---

## Project Structure

```
FairTest_LLM/
├── fairtestllm.py          ← main framework (do not edit)
├── run.py                  ← entry point (edit settings here)
├── bank-full-biased.csv    ← biased bank dataset
├── adult_biased.csv        ← biased adult dataset
├── german_credit_biased.csv← biased german credit dataset
└── results/      ← output folder (auto-created)
    ├── discriminatory_pairs.csv
    ├── test_dataset_model_ready.csv
    └── fairness_report.txt
```

---

## Supported LLMs

| Provider  | Model                     | Key prefix  |
|-----------|---------------------------|-------------|
| OpenAI    | `gpt-4o-mini`, `gpt-4o`   | `sk-...`    |
| Anthropic | `claude-sonnet-4-6`       | `sk-ant-...`|

---

## Supported ML Models

| Code  | Model               |
|-------|---------------------|
| `rf`  | Random Forest       |
| `lr`  | Logistic Regression |
| `mlp` | MLP Neural Network  |

---

## Quick Start

Edit `run.py` with your settings:

```python
agent = FairAgent(
    dataset_path      = "bank-full-biased.csv",
    target_col        = "y",
    api_key           = "sk-your-key-here",   # ← your API key
    llm_model         = "gpt-4o-mini",        # ← LLM model
    model_type        = "rf",                 # ← ML model
    protected_attrs   = ["age", "marital"],   # ← protected attrs
    n_seeds           = 30,
    n_counterfactuals = 8,
    max_retries       = 0,
    n_iterations      = 3,
    fairness_threshold= -1.0,                 # always search
    output_dir        = "results"
)
```

Then run:

```bash
python run.py
```

---

## Datasets

| Dataset       | Rows   | Target        | Protected Attrs              |
|---------------|--------|---------------|------------------------------|
| Bank Marketing| 45,211 | `y`           | `age`, `marital`             |
| Adult Income  | 32,561 | `income`      | `sex`, `race`                |
| German Credit | 1,000  | `credit_risk` | `age`, `personal_status_sex` |

---

## Fairness Metric

| Metric | Fair Value | Meaning                       |
|--------|------------|-------------------------------|
| SPD    | 0.0        | Statistical Parity Difference |

SPD = P(Ŷ=1 | privileged) − P(Ŷ=1 | unprivileged).
A value of 0 indicates perfect fairness. Positive values
indicate the privileged group receives more favourable
predictions.

---

## Output

```
Done! Found 189 discriminatory pairs
G=189 F=243 U=0
Cost       : $0.045
Total time : 312.4s (5.2 min)
```

Results saved to `results/`.
