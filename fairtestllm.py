# -*- coding: utf-8 -*-
"""
FairAgent: Agentic Framework for ML Fairness Testing
Domain: Bank Marketing / Financial
LLM: GPT-4o-mini (default, configurable)
Inspired by CoverUp's coverage-guided iterative LLM approach

User provides:
  - Dataset path (CSV)
  - Target column
  - API key
  - Optionally: protected attributes, model type

Outputs:
  1. discriminatory_pairs.csv        → full pairs with metadata
  2. test_dataset_model_ready.csv    → run directly against model
  3. how_to_use_test_dataset.py      → usage code
  4. fairness_report.txt             → detailed metrics report
  5. visualizations/                 → 4 charts
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Any

warnings.filterwarnings('ignore')

try:
    from openai import OpenAI
    try:
        from anthropic import Anthropic
    except ImportError:
        Anthropic = None
    try:
        from groq import Groq
    except ImportError:
        Groq = None
except ImportError:
    raise ImportError(
        "Please install openai:\n  pip install openai"
    )

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
except ImportError:
    raise ImportError(
        "Please install scikit-learn:\n  pip install scikit-learn"
    )


# ─────────────────────────────────────────────────────────────
# CONSTANTS: Known protected attributes across domains
# ─────────────────────────────────────────────────────────────

KNOWN_PROTECTED = [
    "age", "gender", "sex", "race", "ethnicity",
    "marital", "marital_status", "religion",
    "nationality", "disability", "education",
    "education_level"
]

SUPPORTED_MODELS = {
    "1": ("Random Forest",          "rf"),
    "2": ("Logistic Regression",    "lr"),
    "3": ("Gradient Boosting",      "gb"),
    "4": ("Support Vector Machine",  "svm"),
    "5": ("MLP Neural Network",      "mlp"),
}


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class FairnessMetrics:
    spd: float = 0.0
    eod: float = 0.0
    di:  float = 1.0
    aod: float = 0.0
    attribute: str = ""

    def is_biased(self, threshold: float = 0.1) -> bool:
        return (
            abs(self.spd) > threshold or
            abs(self.eod) > threshold or
            self.di < 0.8 or
            abs(self.aod) > threshold
        )

    def summary(self) -> str:
        return (
            f"SPD={self.spd:+.3f} | "
            f"EOD={self.eod:+.3f} | "
            f"DI={self.di:.3f} | "
            f"AOD={self.aod:+.3f}"
        )

    def bias_level(self) -> str:
        score = (
            abs(self.spd) + abs(self.eod) +
            abs(self.aod) + max(0, 1 - self.di)
        ) / 4
        if score > 0.3:   return "SEVERE [FAIL]"
        elif score > 0.2: return "HIGH [WARN]"
        elif score > 0.1: return "MODERATE [WARN]"
        else:             return "LOW [OK]"


@dataclass
class DiscriminatoryPair:
    original: dict
    counterfactual: dict
    original_prediction: Any
    counterfactual_prediction: Any
    original_probability: float
    counterfactual_probability: float
    changed_attributes: list
    bias_type: str
    explanation: str
    metric_change: float
    attribute: str


@dataclass
class AgentState:
    good: int = 0
    failed: int = 0
    useless: int = 0
    retries: int = 0
    total_cost: float = 0.0
    discriminatory_pairs: list = field(default_factory=list)

    def log(self):
        print(
            f"  G={self.good} (found) | "
            f"F={self.failed} (not found) | "
            f"U={self.useless} (invalid) | "
            f"R={self.retries} (retries) | "
            f"cost=~${self.total_cost:.3f}"
        )


# ─────────────────────────────────────────────────────────────
# DATASET LOADER & PREPROCESSOR
# ─────────────────────────────────────────────────────────────

class DatasetLoader:
    """
    Loads and preprocesses any CSV dataset
    Handles encoding of categorical variables
    """

    # UCI Bank Marketing column names (no header in file)
    BANK_COLS = [
        'age', 'job', 'marital', 'education', 'default',
        'balance', 'housing', 'loan', 'contact', 'day',
        'month', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'y'
    ]

    def __init__(self, path: str, target_col: str):
        self.path       = path
        self.target_col = target_col
        self.encoders   = {}
        self.raw_df     = None
        self.encoded_df = None
        self.feature_names = []

    def load(self) -> pd.DataFrame:
        """Load CSV - auto detects separator and header"""
        print(f"\n[LOAD] Loading dataset: {self.path}")

        # Try with header first
        try:
            df = pd.read_csv(self.path, sep=';')
            if self.target_col in df.columns:
                self.raw_df = df
                print(f"  Loaded with ';' separator and header")
                print(f"  Shape: {df.shape}")
                return df
        except Exception:
            pass

        # Try comma separator with header
        try:
            df = pd.read_csv(self.path, sep=',')
            if self.target_col in df.columns:
                self.raw_df = df
                print(f"  Loaded with ',' separator and header")
                print(f"  Shape: {df.shape}")
                return df
        except Exception:
            pass

        # Try bank-full.csv format (no header, semicolon)
        try:
            df = pd.read_csv(
                self.path, sep=';', names=self.BANK_COLS
            )
            if self.target_col in df.columns:
                self.raw_df = df
                print(f"  Loaded as UCI Bank dataset (no header)")
                print(f"  Shape: {df.shape}")
                return df
        except Exception:
            pass

        raise ValueError(
            f"Could not load dataset from {self.path}\n"
            f"Make sure target column '{self.target_col}' exists."
        )

    def encode(self) -> pd.DataFrame:
        """
        Encode categorical columns for ML model
        Keeps original values for LLM prompting
        """
        df = self.raw_df.copy()

        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == 'str':
                le = LabelEncoder()
                df[col] = le.fit_transform(
                    df[col].astype(str)
                )
                self.encoders[col] = le

        self.encoded_df = df
        self.feature_names = [
            c for c in df.columns
            if c != self.target_col
        ]
        return df

    def get_column_info(self) -> dict:
        """Get info about each column for LLM context"""
        info = {}
        for col in self.raw_df.columns:
            if col == self.target_col:
                continue
            col_data = self.raw_df[col]
            if col_data.dtype in ['int64', 'float64']:
                uniq = sorted(col_data.unique().tolist())
                info[col] = {
                    "type": "numeric",
                    "min":  float(col_data.min()),
                    "max":  float(col_data.max()),
                    "mean": round(float(col_data.mean()), 2),
                    "unique_values": (
                        uniq if len(uniq) <= 15 else None
                    )
                }
            else:
                info[col] = {
                    "type": "categorical",
                    "unique_values": col_data.unique().tolist()
                }
        return info

    def decode_sample(self, encoded_sample: dict) -> dict:
        """Convert encoded values back to original labels"""
        decoded = {}
        for col, val in encoded_sample.items():
            if col in self.encoders:
                try:
                    decoded[col] = self.encoders[col].inverse_transform(
                        [int(val)]
                    )[0]
                except Exception:
                    decoded[col] = val
            else:
                decoded[col] = val
        return decoded

    def encode_sample(self, raw_sample: dict) -> dict:
        """Convert raw values to encoded values for model"""
        encoded = {}
        for col, val in raw_sample.items():
            if col not in self.feature_names and col != self.target_col:
                continue
            if col in self.encoders:
                try:
                    # Try direct transform
                    encoded[col] = int(
                        self.encoders[col].transform([str(val)])[0]
                    )
                except Exception:
                    try:
                        # Try finding closest class
                        classes = self.encoders[col].classes_
                        val_str = str(val).lower().strip()
                        match = None
                        for cls in classes:
                            if cls.lower() == val_str:
                                match = cls
                                break
                        if match:
                            encoded[col] = int(
                                self.encoders[col].transform([match])[0]
                            )
                        else:
                            # Use most frequent class as fallback
                            encoded[col] = int(
                                self.encoders[col].transform(
                                    [classes[0]]
                                )[0]
                            )
                    except Exception:
                        encoded[col] = 0
            else:
                try:
                    encoded[col] = float(val)
                except Exception:
                    encoded[col] = 0
        return encoded


# ─────────────────────────────────────────────────────────────
# MODEL TRAINER
# ─────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains ML model on the dataset
    User can choose model type or provide their own
    """

    def __init__(
        self,
        model_type: str = "rf",
        test_size: float = 0.2
    ):
        self.model_type = model_type
        self.test_size  = test_size
        self.model      = None
        self.X_test     = None
        self.y_test     = None

    def build_model(self):
        """Build the selected model"""
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        elif self.model_type == "lr":
            return LogisticRegression(
                max_iter=1000, random_state=42
            )
        elif self.model_type == "svm":
            return SVC(
                kernel="rbf", random_state=42,
                probability=True
            )
        elif self.model_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=500,
                random_state=42
            )
        elif self.model_type == "gb":
            return GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )
        else:
            return RandomForestClassifier(
                n_estimators=100, random_state=42
            )

    def train(
        self,
        encoded_df: pd.DataFrame,
        target_col: str
    ):
        """Train model and print performance"""
        print(f"\n[MODEL] Training model ({self.model_type})...")

        X = encoded_df.drop(columns=[target_col])
        y = encoded_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        self.model  = self.build_model()
        self.model.fit(X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test

        # Print performance
        y_pred   = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  [OK] Model trained successfully!")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Test samples:     {len(X_test):,}")

        return self.model


# ─────────────────────────────────────────────────────────────
# PROTECTED ATTRIBUTE DETECTOR
# ─────────────────────────────────────────────────────────────

class ProtectedAttrDetector:
    """
    Detects potential protected attributes from column names
    Suggests them to user for confirmation
    """

    def detect_and_confirm(
        self,
        columns: list,
        target_col: str
    ) -> list:
        """
        Auto-detect protected attributes and ask user to confirm
        """
        cols_lower = {
            c.lower(): c for c in columns
            if c != target_col
        }

        suggested = []
        for known in KNOWN_PROTECTED:
            for col_lower, col_orig in cols_lower.items():
                if known in col_lower:
                    if col_orig not in suggested:
                        suggested.append(col_orig)

        print(f"\n[SEARCH] Detected potential protected attributes:")
        if not suggested:
            print("  None auto-detected.")
            print(
                "  Available columns: "
                f"{[c for c in columns if c != target_col]}"
            )
            return self._manual_input(columns, target_col)

        confirmed = []
        for attr in suggested:
            resp = input(
                f"  Include '{attr}' as protected attribute? "
                f"[y/n]: "
            ).strip().lower()
            if resp == 'y':
                confirmed.append(attr)

        # Ask if user wants to add more
        resp = input(
            "\n  Add any more protected attributes manually? "
            "[y/n]: "
        ).strip().lower()
        if resp == 'y':
            extra = self._manual_input(
                columns, target_col, existing=confirmed
            )
            confirmed.extend(extra)

        if not confirmed:
            print(
                "  [WARN]  No protected attributes selected. "
                "Please select at least one."
            )
            confirmed = self._manual_input(
                columns, target_col
            )

        return confirmed

    def _manual_input(
        self,
        columns: list,
        target_col: str,
        existing: list = None
    ) -> list:
        """Let user manually enter protected attributes"""
        existing = existing or []
        available = [
            c for c in columns
            if c != target_col and c not in existing
        ]
        print(f"  Available columns: {available}")
        raw = input(
            "  Enter protected attributes "
            "(comma-separated): "
        ).strip()
        attrs = [
            a.strip() for a in raw.split(',')
            if a.strip() in available
        ]
        return attrs


# ─────────────────────────────────────────────────────────────
# FAIRNESS METRICS CALCULATOR
# ─────────────────────────────────────────────────────────────

class FairnessCalculator:
    """
    Like CoverUp's SlipCover - measures fairness coverage
    """

    def __init__(
        self,
        model,
        encoded_df: pd.DataFrame,
        protected_attrs: list,
        target_col: str
    ):
        self.model           = model
        self.encoded_df      = encoded_df
        self.protected_attrs = protected_attrs
        self.target_col      = target_col

    def measure(
        self,
        attr: str,
        privileged_val: Any,
        unprivileged_val: Any
    ) -> FairnessMetrics:
        """Measure fairness metrics for one protected attribute"""
        X      = self.encoded_df.drop(columns=[self.target_col])
        y_true = self.encoded_df[self.target_col].values
        y_pred = self.model.predict(X)

        priv_mask   = (self.encoded_df[attr] == privileged_val).values
        unpriv_mask = (self.encoded_df[attr] == unprivileged_val).values

        if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
            return FairnessMetrics(attribute=attr)

        priv_pred   = y_pred[priv_mask]
        unpriv_pred = y_pred[unpriv_mask]
        priv_true   = y_true[priv_mask]
        unpriv_true = y_true[unpriv_mask]

        # SPD
        spd = float(priv_pred.mean() - unpriv_pred.mean())

        # DI
        di = (
            float(unpriv_pred.mean() / priv_pred.mean())
            if priv_pred.mean() > 0 else 1.0
        )

        # EOD
        priv_tpr = (
            float(
                ((priv_pred == 1) & (priv_true == 1)).sum() /
                (priv_true == 1).sum()
            ) if (priv_true == 1).sum() > 0 else 0.0
        )
        unpriv_tpr = (
            float(
                ((unpriv_pred == 1) & (unpriv_true == 1)).sum() /
                (unpriv_true == 1).sum()
            ) if (unpriv_true == 1).sum() > 0 else 0.0
        )
        eod = priv_tpr - unpriv_tpr

        # AOD
        priv_fpr = (
            float(
                ((priv_pred == 1) & (priv_true == 0)).sum() /
                (priv_true == 0).sum()
            ) if (priv_true == 0).sum() > 0 else 0.0
        )
        unpriv_fpr = (
            float(
                ((unpriv_pred == 1) & (unpriv_true == 0)).sum() /
                (unpriv_true == 0).sum()
            ) if (unpriv_true == 0).sum() > 0 else 0.0
        )
        aod = (
            (priv_fpr - unpriv_fpr) +
            (priv_tpr - unpriv_tpr)
        ) / 2

        return FairnessMetrics(
            spd=round(spd, 4),
            eod=round(eod, 4),
            di=round(min(di, 2.0), 4),
            aod=round(aod, 4),
            attribute=attr
        )


# ─────────────────────────────────────────────────────────────
# SHAP EXPLAINER
# ─────────────────────────────────────────────────────────────

class SHAPExplainer:
    """Like CoverUp's get_info tool - gives LLM feature context"""

    def __init__(
        self,
        model,
        encoded_df: pd.DataFrame,
        target_col: str
    ):
        self.model        = model
        self.target_col   = target_col
        X                 = encoded_df.drop(columns=[target_col])
        self.feature_names = list(X.columns)
        self.background   = X.sample(
            min(50, len(X)), random_state=42
        )

    def get_global_importance(self) -> dict:
        """Global feature importance"""
        try:
            import shap
            exp   = shap.TreeExplainer(self.model)
            vals  = exp.shap_values(self.background)
            if isinstance(vals, list):
                vals = vals[1]
            imp = dict(zip(
                self.feature_names,
                np.abs(vals).mean(axis=0).tolist()
            ))
            return dict(sorted(
                imp.items(), key=lambda x: x[1], reverse=True
            ))
        except Exception:
            try:
                imp = dict(zip(
                    self.feature_names,
                    self.model.feature_importances_.tolist()
                ))
                return dict(sorted(
                    imp.items(),
                    key=lambda x: x[1], reverse=True
                ))
            except Exception:
                return {f: 1.0 for f in self.feature_names}

    def get_local_importance(self, sample_encoded: dict) -> dict:
        """Local feature importance for one sample"""
        try:
            import shap
            exp  = shap.TreeExplainer(self.model)
            sdf  = pd.DataFrame([sample_encoded])[
                self.feature_names
            ]
            vals = exp.shap_values(sdf)
            if isinstance(vals, list):
                vals = vals[1]
            return dict(zip(
                self.feature_names, vals[0].tolist()
            ))
        except Exception:
            return {f: 0.0 for f in self.feature_names}


# ─────────────────────────────────────────────────────────────
# LLM INTERFACE
# ─────────────────────────────────────────────────────────────

class LLMInterface:
    """Like CoverUp's LLM chat interface"""

    # ── Cost per 1M tokens ────────────────────────────────────
    COSTS = {
        # OpenAI models
        "gpt-4o-mini":               {"input": 0.15,  "output": 0.60},
        "gpt-4o":                    {"input": 2.50,  "output": 10.00},
        "gpt-4-turbo":               {"input": 10.00, "output": 30.00},
        # Claude models (current 2025)
        "claude-haiku-4-5-20251001": {"input": 1.00,  "output": 5.00},
        "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00},
        "claude-opus-4-6":           {"input": 15.00, "output": 75.00},
        # Claude models (older)
        "claude-3-5-haiku-20241022": {"input": 0.80,  "output": 4.00},
        "claude-3-5-sonnet-20241022":{"input": 3.00,  "output": 15.00},
        # Groq models (free tier available)
        "llama-3.3-70b-versatile":   {"input": 0.59,  "output": 0.79},
        "llama-3.1-8b-instant":      {"input": 0.05,  "output": 0.08},
        "mixtral-8x7b-32768":        {"input": 0.24,  "output": 0.24},
        "gemma2-9b-it":              {"input": 0.20,  "output": 0.20},
    }

    # Claude model names for detection
    CLAUDE_MODELS = {
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    }

    # Groq model names for detection
    GROQ_MODELS = {
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "gemma-7b-it",
    }

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.model      = model
        self.total_cost = 0.0

        # ── CHANGED: choose client based on model ──────────────
        if model in self.CLAUDE_MODELS or "claude" in model.lower():
            if Anthropic is None:
                raise ImportError(
                    "Please install anthropic:\n  pip install anthropic"
                )
            self.client    = Anthropic(api_key=api_key)
            self.is_claude = True
            self.is_groq   = False
            print(f"  [LLM] Using Claude API: {model}")

        elif model in self.GROQ_MODELS or "llama" in model.lower()                 or "mixtral" in model.lower() or "gemma" in model.lower():
            if Groq is None:
                raise ImportError(
                    "Please install groq:\n  pip install groq"
                )
            self.client    = Groq(api_key=api_key)
            self.is_claude = False
            self.is_groq   = True
            print(f"  [LLM] Using Groq API: {model}")

        else:
            self.client    = OpenAI(api_key=api_key)
            self.is_claude = False
            self.is_groq   = False
            print(f"  [LLM] Using OpenAI API: {model}")
        # ──────────────────────────────────────────────────────

    def chat(
        self,
        messages: list,
        max_tokens: int = 2000
    ) -> str | None:
        """Send messages to LLM — supports both OpenAI and Claude"""
        try:
            if self.is_claude:
                # ── Claude API ─────────────────────────────────
                system_msg = ""
                user_msgs  = []
                for m in messages:
                    if m["role"] == "system":
                        system_msg = m["content"]
                    else:
                        user_msgs.append(m)

                resp = self.client.messages.create(
                    model      = self.model,
                    max_tokens = max_tokens,
                    system     = system_msg,
                    messages   = user_msgs
                )
                self.total_cost += self._cost(
                    resp.usage.input_tokens,
                    resp.usage.output_tokens
                )
                return resp.content[0].text

            elif self.is_groq:
                # ── Groq API (same interface as OpenAI) ────────
                resp = self.client.chat.completions.create(
                    model       = self.model,
                    messages    = messages,
                    max_tokens  = max_tokens,
                    temperature = 0.7
                )
                self.total_cost += self._cost(
                    resp.usage.prompt_tokens,
                    resp.usage.completion_tokens
                )
                return resp.choices[0].message.content

            else:
                # ── OpenAI API ─────────────────────────────────
                resp = self.client.chat.completions.create(
                    model       = self.model,
                    messages    = messages,
                    max_tokens  = max_tokens,
                    temperature = 0.7
                )
                self.total_cost += self._cost(
                    resp.usage.prompt_tokens,
                    resp.usage.completion_tokens
                )
                return resp.choices[0].message.content

        except Exception as e:
            print(f"\n  LLM Error: {e}")
            return None

    def _cost(self, inp: int, out: int) -> float:
        r = self.COSTS.get(
            self.model, {"input": 0.15, "output": 0.60}
        )
        return (
            (inp / 1_000_000) * r["input"] +
            (out / 1_000_000) * r["output"]
        )


# ─────────────────────────────────────────────────────────────
# GLOBAL SEARCH AGENT
# ─────────────────────────────────────────────────────────────

class GlobalSearchAgent:
    """
    PHASE 1: Global Search
    LLM reasons about entire dataset, generates diverse seeds
    Like CoverUp's initial prompting across all segments
    """

    SYSTEM = """You are an expert ML fairness auditor
specializing in financial/banking discrimination.
Find discrimination in ML models by generating
patient/customer profiles that expose bias.
Respond ONLY with valid JSON arrays. No extra text."""

    def __init__(
        self,
        llm: LLMInterface,
        loader: DatasetLoader,
        protected_attrs: list,
        target_col: str
    ):
        self.llm             = llm
        self.loader          = loader
        self.protected_attrs = protected_attrs
        self.target_col      = target_col
        self.col_info        = loader.get_column_info()
        self.feature_names   = loader.feature_names

    def generate_seeds(
        self,
        metrics: FairnessMetrics,
        shap_importance: dict,
        n_seeds: int = 20
    ) -> list:
        """
        Generate diverse seed samples using LLM
        Returns samples in RAW (original) format
        """
        top_shap = dict(list(shap_importance.items())[:8])

        prompt = f"""
You are auditing a bank marketing ML model for discrimination.
The model predicts if a customer will subscribe to a term deposit.

CURRENT FAIRNESS METRICS (0 = perfectly fair):
- SPD = {metrics.spd} (positive = privileged group subscribes more)
- EOD = {metrics.eod}
- DI  = {metrics.di} (below 0.8 = legally problematic)
- AOD = {metrics.aod}
- Bias Level: {metrics.bias_level()}

TOP FEATURE IMPORTANCE (SHAP):
{json.dumps(top_shap, indent=2)}

PROTECTED ATTRIBUTES TO TEST: {self.protected_attrs}

DATASET COLUMN INFO (use ONLY these exact values):
{json.dumps(self.col_info, indent=2)}

TASK: Generate {n_seeds} diverse customer profiles as seeds.
Focus on:
1. Different demographic combinations
2. Intersectional cases (e.g., older + divorced + primary education)
3. Borderline customers where bias could tip the decision
4. Realistic financial profiles

Use ONLY the exact values shown in column info above.
Features needed: {self.feature_names}

Respond with ONLY a JSON array:
[
  {{"feature1": value1, "feature2": value2, ...}},
  ...
]"""

        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user",   "content": prompt}
        ]

        # ── CHANGED: cap tokens to avoid truncated JSON ──────
        # Each seed ≈ 90 tokens but cap at 4000 to prevent
        # incomplete JSON responses from Claude
        dynamic_tokens = min(4000, max(2000, n_seeds * 90 + 500))
        response = self.llm.chat(messages, max_tokens=dynamic_tokens)
        if not response:
            return []

        try:
            raw = response
            raw = raw.replace("```json","").replace("```","").strip()

            # ── CHANGED: parse individual objects if full array fails ──
            # Claude truncates long responses → no closing ]
            # Solution: extract each {...} object individually
            valid = []
            import re

            # Try full array first
            s = raw.find('[')
            e = raw.rfind(']') + 1
            if s >= 0 and e > s:
                try:
                    seeds = json.loads(raw[s:e])
                    for seed in seeds:
                        if isinstance(seed, dict):
                            for f in self.feature_names:
                                if f not in seed:
                                    seed[f] = 0
                            valid.append(seed)
                    if valid:
                        return valid[:n_seeds]
                except json.JSONDecodeError:
                    pass

            # Fallback: extract each { ... } object individually
            # Works even when response is truncated
            depth   = 0
            start   = -1
            for i, ch in enumerate(raw):
                if ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start >= 0:
                        obj_str = raw[start:i+1]
                        try:
                            obj = json.loads(obj_str)
                            if isinstance(obj, dict):
                                matching = sum(
                                    1 for f in self.feature_names
                                    if f in obj
                                )
                                if matching >= len(
                                    self.feature_names
                                ) * 0.5:
                                    for f in self.feature_names:
                                        if f not in obj:
                                            obj[f] = 0
                                    valid.append(obj)
                                    if len(valid) >= n_seeds:
                                        break
                        except json.JSONDecodeError:
                            pass
            # ──────────────────────────────────────────────────────

            return valid[:n_seeds]

        except Exception as e:
            print(f"  Warning: seed parsing error: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# LOCAL SEARCH AGENT
# ─────────────────────────────────────────────────────────────

class LocalSearchAgent:
    """
    PHASE 2: Local Search
    LLM generates counterfactuals by changing protected attrs only
    Like CoverUp's continued chat for specific segments
    """

    SYSTEM = """You are an expert ML fairness auditor.
Generate counterfactual customer profiles to test discrimination.
Rule: change ONLY protected attributes, keep ALL other features identical.
This tests if same customer gets different outcome based on demographics.
Respond ONLY with valid JSON. No extra text."""

    def __init__(
        self,
        llm: LLMInterface,
        loader: DatasetLoader,
        protected_attrs: list
    ):
        self.llm             = llm
        self.loader          = loader
        self.protected_attrs = protected_attrs
        self.col_info        = loader.get_column_info()
        self.feature_names   = loader.feature_names
        self.non_protected   = [
            f for f in self.feature_names
            if f not in protected_attrs
        ]

    def generate_counterfactuals(
        self,
        seed_raw: dict,
        local_shap: dict,
        metrics: FairnessMetrics,
        n: int = 10
    ) -> list:
        """
        Generate counterfactual customers around a seed
        Like CoverUp's local search
        """
        # Protected attribute possible values
        attr_vals = {
            attr: self.col_info.get(attr, {}).get(
                "unique_values", []
            )
            for attr in self.protected_attrs
        }

        top_shap = dict(
            sorted(
                local_shap.items(),
                key=lambda x: abs(np.mean(x[1])) if isinstance(x[1], (list, np.ndarray)) else abs(x[1]), reverse=True
            )[:6]
        )

        prompt = f"""
Testing this bank customer for discriminatory prediction:

ORIGINAL CUSTOMER:
{json.dumps(seed_raw, indent=2)}

LOCAL SHAP (feature influence on this customer's prediction):
{json.dumps(top_shap, indent=2)}

BIAS METRICS: {metrics.summary()}

PROTECTED ATTRIBUTES - ONLY these can change:
{json.dumps(attr_vals, indent=2)}

NON-PROTECTED FEATURES - must stay EXACTLY the same:
{self.non_protected}

TASK: Generate {n} counterfactual customers:
1. Keep ALL non-protected features IDENTICAL
2. Change ONLY protected attributes (age/marital/education)
3. Cover all possible combinations
4. Include intersectional changes (multiple protected attrs)
5. Use ONLY exact values from the data

Tests: "Would same customer get different bank outcome
if they had different demographic characteristics?"

Respond with ONLY a JSON array:
[
  {{
    "changed_attrs": ["attr1"],
    "sample": {{...complete customer profile...}}
  }},
  ...
]"""

        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user",   "content": prompt}
        ]




        # Scale max_tokens with n_seeds to avoid truncation
        dynamic_tokens = min(16000, max(3000, n * 90 + 500))
        response = self.llm.chat(messages, max_tokens=dynamic_tokens)
        if not response:
            return []

        try:
            s = response.find('[')
            e = response.rfind(']') + 1
            if s >= 0 and e > s:
                cfs = json.loads(response[s:e])
                valid = [
                    cf for cf in cfs
                    if isinstance(cf, dict) and
                    "sample" in cf and
                    "changed_attrs" in cf and
                    all(
                        f in cf["sample"]
                        for f in self.feature_names
                    )
                ]
                return valid[:n]
        except json.JSONDecodeError:
            print("\n  Warning: Could not parse counterfactuals")
        return []

    def retry_with_feedback(
        self,
        seed_raw: dict,
        failed_cf: dict,
        metrics: FairnessMetrics
    ) -> list:
        """
        Like CoverUp's continued chat after failure
        Tells LLM what failed and asks to try differently
        """
        attr_vals = {
            attr: self.col_info.get(attr, {}).get(
                "unique_values", []
            )
            for attr in self.protected_attrs
        }

        prompt = f"""
Previous attempt FAILED - same prediction for both customers.

ORIGINAL: {json.dumps(seed_raw, indent=2)}
FAILED COUNTERFACTUAL: {json.dumps(
    failed_cf.get('sample', {}), indent=2
)}

BIAS STILL EXISTS: {metrics.summary()}
Discrimination is real but not found at this point yet.

Available protected values: {json.dumps(attr_vals, indent=2)}

Try a DIFFERENT approach:
- Different combinations of protected attributes
- More extreme demographic changes
- Intersectional changes (multiple protected attrs together)

Keep ALL non-protected features identical.
Respond with ONLY a JSON array of 5 new counterfactuals:
[
  {{"changed_attrs": ["x"], "sample": {{...}}}},
  ...
]"""

        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user",   "content": prompt}
        ]

        response = self.llm.chat(messages, max_tokens=2000)
        if not response:
            return []

        try:
            s = response.find('[')
            e = response.rfind(']') + 1
            if s >= 0 and e > s:
                return json.loads(response[s:e])
        except json.JSONDecodeError:
            pass
        return []


# ─────────────────────────────────────────────────────────────
# VISUALIZER
# ─────────────────────────────────────────────────────────────

class Visualizer:
    """Generate all charts for fairness analysis"""

    def __init__(self, output_dir: str):
        self.out = os.path.join(output_dir, "visualizations")
        os.makedirs(self.out, exist_ok=True)
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            plt.style.use('ggplot')

    def plot_fairness_metrics(self, all_metrics: dict):
        attrs = list(all_metrics.keys())
        spds  = [all_metrics[a].spd for a in attrs]
        eods  = [all_metrics[a].eod for a in attrs]
        aods  = [all_metrics[a].aod for a in attrs]
        dis   = [all_metrics[a].di  for a in attrs]

        x = np.arange(len(attrs))
        w = 0.25

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # SPD / EOD / AOD
        ax = axes[0]
        ax.bar(x - w,  spds, w, label='SPD', color='#E74C3C', alpha=0.8)
        ax.bar(x,      eods, w, label='EOD', color='#3498DB', alpha=0.8)
        ax.bar(x + w,  aods, w, label='AOD', color='#9B59B6', alpha=0.8)
        ax.axhline(y=0.1,  color='orange', linestyle='--',
                   lw=1.5, label='Bias threshold')
        ax.axhline(y=-0.1, color='orange', linestyle='--', lw=1.5)
        ax.axhline(y=0,    color='green',  linestyle='-',
                   lw=1.5, label='Fair (0.0)')
        ax.set_xticks(x)
        ax.set_xticklabels(attrs, fontsize=11)
        ax.set_ylabel('Score (closer to 0 = fairer)')
        ax.set_title('SPD / EOD / AOD Metrics', fontsize=13,
                     fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim(-0.7, 0.7)

        # Disparate Impact
        ax2 = axes[1]
        colors = [
            '#E74C3C' if d < 0.8 else '#2ECC71' for d in dis
        ]
        bars = ax2.bar(attrs, dis, color=colors, alpha=0.8, width=0.4)
        ax2.axhline(y=0.8, color='red',   linestyle='--',
                    lw=2, label='Legal threshold (0.8)')
        ax2.axhline(y=1.0, color='green', linestyle='-',
                    lw=2, label='Perfect fairness (1.0)')
        for bar, val in zip(bars, dis):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10
            )
        ax2.set_ylabel('Disparate Impact')
        ax2.set_title('Disparate Impact (≥ 0.8 required)',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.set_ylim(0, 1.5)

        fig.suptitle(
            'Bank Marketing Model - Fairness Metrics',
            fontsize=15, fontweight='bold'
        )
        plt.tight_layout()
        path = os.path.join(self.out, "fairness_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] {path}")

    def plot_discrimination_heatmap(self, pairs: list):
        if not pairs:
            return
        bias_counts = {}
        for p in pairs:
            k = " + ".join(sorted(p.changed_attributes))
            bias_counts[k] = bias_counts.get(k, 0) + 1

        labels = list(bias_counts.keys())
        counts = list(bias_counts.values())

        fig, ax = plt.subplots(figsize=(10, 5))
        colors  = plt.cm.Reds(
            np.linspace(0.3, 0.9, len(counts))
        )
        bars = ax.barh(labels, counts, color=colors)
        for bar, c in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.2,
                bar.get_y() + bar.get_height() / 2,
                str(c), va='center', fontsize=11,
                fontweight='bold'
            )
        ax.set_xlabel('Discriminatory Pairs Found')
        ax.set_title(
            'Discrimination by Attribute Combination',
            fontsize=13, fontweight='bold'
        )
        ax.set_xlim(0, max(counts) * 1.15)
        plt.tight_layout()
        path = os.path.join(
            self.out, "discrimination_heatmap.png"
        )
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] {path}")

    def plot_prediction_distribution(self, pairs: list):
        if not pairs:
            return
        orig    = [p.original_probability    for p in pairs]
        counter = [p.counterfactual_probability for p in pairs]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(orig, counter, alpha=0.5,
                   c='#E74C3C', s=30)
        ax.plot([0,1],[0,1],'b--', lw=1.5,
                label='No difference')
        ax.set_xlabel('Original Probability')
        ax.set_ylabel('Counterfactual Probability')
        ax.set_title('Prediction Probability Comparison',
                     fontsize=12)
        ax.legend()

        ax2 = axes[1]
        diffs = [abs(o - c) for o, c in zip(orig, counter)]
        ax2.hist(diffs, bins=20, color='#E74C3C',
                 alpha=0.7, edgecolor='black')
        ax2.axvline(
            np.mean(diffs), color='blue',
            linestyle='--', lw=2,
            label=f'Mean={np.mean(diffs):.3f}'
        )
        ax2.set_xlabel('Absolute Probability Difference')
        ax2.set_ylabel('Count')
        ax2.set_title('Discrimination Severity Distribution',
                      fontsize=12)
        ax2.legend()

        fig.suptitle(
            'Discriminatory Prediction Analysis',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        path = os.path.join(
            self.out, "prediction_distribution.png"
        )
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] {path}")

    def plot_agent_progress(self, state: AgentState):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Pie
        ax    = axes[0]
        total = state.good + state.failed + state.useless
        if total > 0:
            sizes  = [state.good, state.failed, state.useless]
            labels = [
                f'Found (G)\n{state.good}',
                f'Not Found (F)\n{state.failed}',
                f'Invalid (U)\n{state.useless}'
            ]
            colors  = ['#2ECC71', '#E74C3C', '#F39C12']
            explode = (0.05, 0, 0)
            ax.pie(
                sizes, labels=labels, colors=colors,
                explode=explode, autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 11}
            )
            ax.set_title(
                'Agent Search Results (CoverUp G/F/U style)',
                fontsize=12, fontweight='bold'
            )

        # Bar summary
        ax2 = axes[1]
        names  = ['Found (G)', 'Failed (F)',
                  'Invalid (U)', 'Retries (R)']
        vals   = [
            state.good, state.failed,
            state.useless, state.retries
        ]
        colors2 = ['#2ECC71','#E74C3C','#F39C12','#3498DB']
        bars = ax2.bar(names, vals, color=colors2, alpha=0.8)
        for bar, v in zip(bars, vals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(v), ha='center',
                fontsize=12, fontweight='bold'
            )
        ax2.set_ylabel('Count')
        ax2.set_title(
            f'Testing Summary | Cost: ${state.total_cost:.3f}',
            fontsize=12, fontweight='bold'
        )

        plt.tight_layout()
        path = os.path.join(self.out, "agent_progress.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] {path}")


# ─────────────────────────────────────────────────────────────
# MAIN FAIRAGENT CLASS
# ─────────────────────────────────────────────────────────────

class FairAgent:
    """
    Main Agentic Framework for ML Fairness Testing
    Bank Marketing Domain

    Mapping to CoverUp:
    ┌──────────────────────────────────────────────┐
    │  CoverUp              →  FairAgent           │
    │  ──────────────────────────────────────────  │
    │  Measure coverage     →  Measure fairness    │
    │  Find uncovered lines →  Find biased groups  │
    │  Global segments      →  Global LLM search   │
    │  Local prompting      →  Local LLM search    │
    │  G/F/U/R tracking     →  G/F/U/R tracking   │
    │  Continued chat       →  Feedback loop       │
    │  get_info tool        →  SHAP explainer      │
    │  Test suite           →  Discriminatory pairs│
    └──────────────────────────────────────────────┘

    Minimum usage:
        agent = FairAgent(
            dataset_path = "bank-full.csv",
            target_col   = "y",
            api_key      = "sk-..."
        )
        agent.run()

    Full usage:
        agent = FairAgent(
            dataset_path     = "bank-full.csv",
            target_col       = "y",
            api_key          = "sk-...",
            llm_model        = "gpt-4o-mini",
            model_type       = "rf",
            protected_attrs  = ["age", "marital"],
            n_seeds          = 20,
            n_counterfactuals= 10,
            n_iterations     = 3,
            output_dir       = "results"
        )
        agent.run()
    """

    def __init__(
        self,
        dataset_path: str,
        target_col: str,
        api_key: str,
        llm_model: str           = "gpt-4o-mini",
        model_type: str          = None,
        model                    = None,
        protected_attrs: list    = None,
        privileged_groups: dict  = None,
        unprivileged_groups: dict= None,
        n_seeds: int             = 20,
        n_counterfactuals: int   = 10,
        max_retries: int         = 2,
        fairness_threshold: float= 0.,
        n_iterations: int        = 3,
        output_dir: str          = "fairagent_results"
    ):
        self.target_col        = target_col
        self.n_seeds           = n_seeds
        self.n_counterfactuals = n_counterfactuals
        self.max_retries       = max_retries
        self.fairness_threshold= fairness_threshold
        self.n_iterations      = n_iterations
        self.output_dir        = output_dir
        self.state             = AgentState()

        os.makedirs(output_dir, exist_ok=True)

        # ── Load dataset ───────────────────────────────────
        self.loader = DatasetLoader(dataset_path, target_col)
        self.loader.load()
        self.loader.encode()

        # ── Detect protected attributes ────────────────────
        if protected_attrs:
            self.protected_attrs = protected_attrs
            print(
                f"\n[OK] Protected attributes set: "
                f"{protected_attrs}"
            )
        else:
            detector = ProtectedAttrDetector()
            self.protected_attrs = (
                detector.detect_and_confirm(
                    list(self.loader.raw_df.columns),
                    target_col
                )
            )

        print(
            f"\n[OK] Using protected attributes: "
            f"{self.protected_attrs}"
        )

        # ── Train or use provided model ────────────────────
        if model is not None:
            self.model = model
            print("\n[OK] Using provided pre-trained model")
        else:
            if model_type is None:
                model_type = self._ask_model_type()
            trainer = ModelTrainer(model_type=model_type)
            self.model = trainer.train(
                self.loader.encoded_df, target_col
            )

        # ── LLM setup ──────────────────────────────────────
        self.llm = LLMInterface(api_key, llm_model)
        print(f"\n[OK] LLM: {llm_model}")

        # ── Auto detect privileged/unprivileged groups ─────
        self.privileged_groups = privileged_groups or {}
        self.unprivileged_groups = unprivileged_groups or {}

        for attr in self.protected_attrs:
            enc_df = self.loader.encoded_df
            if attr not in self.privileged_groups:
                # Most frequent encoded value = privileged
                self.privileged_groups[attr] = int(
                    enc_df[attr].mode()[0]
                )
            if attr not in self.unprivileged_groups:
                other = [
                    v for v in enc_df[attr].unique()
                    if v != self.privileged_groups[attr]
                ]
                self.unprivileged_groups[attr] = int(
                    other[0]
                ) if other else self.privileged_groups[attr]

        # ── Initialize components ──────────────────────────
        self.fair_calc = FairnessCalculator(
            self.model, self.loader.encoded_df,
            self.protected_attrs, target_col
        )
        self.shap = SHAPExplainer(
            self.model, self.loader.encoded_df, target_col
        )
        self.global_agent = GlobalSearchAgent(
            self.llm, self.loader,
            self.protected_attrs, target_col
        )
        self.local_agent = LocalSearchAgent(
            self.llm, self.loader, self.protected_attrs
        )
        self.visualizer = Visualizer(output_dir)

    def _ask_model_type(self) -> str:
        """Let user choose ML model"""
        print("\n[MODEL] Choose ML model to train:")
        for key, (name, code) in SUPPORTED_MODELS.items():
            print(f"  {key}. {name}")
        choice = input("  Enter choice [1/2/3/4/5] (default=1): ").strip()
        return SUPPORTED_MODELS.get(
            choice, SUPPORTED_MODELS["1"]
        )[1]

    # ── PERCEIVE ──────────────────────────────────────────────

    def _measure_fairness(self) -> dict:
        """Like CoverUp's 'Measuring coverage...'"""
        print("\n[METRICS] Measuring fairness...")
        all_metrics = {}
        for attr in self.protected_attrs:
            m = self.fair_calc.measure(
                attr,
                self.privileged_groups[attr],
                self.unprivileged_groups[attr]
            )
            all_metrics[attr] = m
            flag = "[WARN] BIASED" if m.is_biased(
                self.fairness_threshold
            ) else "[OK] OK"
            print(
                f"  {attr:15s}: {m.summary()} "
                f"[{m.bias_level()}] {flag}"
            )
        return all_metrics

    # ── EVALUATE ──────────────────────────────────────────────

    def _evaluate_pair(
        self,
        seed_raw: dict,
        cf_data: dict,
        attr: str,
        metrics: FairnessMetrics
    ) -> str:
        """Like CoverUp's G/F/U test verification"""
        try:
            sample_raw    = cf_data.get("sample", {})
            changed_attrs = cf_data.get("changed_attrs", [])

            if not sample_raw:
                self.state.useless += 1
                return "U"

            if not all(
                f in sample_raw
                for f in self.loader.feature_names
            ):
                self.state.useless += 1
                return "U"

            # Encode both samples for model
            seed_enc   = self.loader.encode_sample(seed_raw)
            sample_enc = self.loader.encode_sample(sample_raw)

            seed_df   = pd.DataFrame([seed_enc])[
                self.loader.feature_names
            ]
            counter_df = pd.DataFrame([sample_enc])[
                self.loader.feature_names
            ]

            seed_pred    = self.model.predict(seed_df)[0]
            counter_pred = self.model.predict(counter_df)[0]

            try:
                seed_prob    = float(
                    self.model.predict_proba(seed_df)[0][1]
                )
                counter_prob = float(
                    self.model.predict_proba(counter_df)[0][1]
                )
            except Exception:
                seed_prob    = float(seed_pred)
                counter_prob = float(counter_pred)

            # DISCRIMINATION FOUND!
            if seed_pred != counter_pred:
                pair = DiscriminatoryPair(
                    original=seed_raw,
                    counterfactual=sample_raw,
                    original_prediction=int(seed_pred),
                    counterfactual_prediction=int(counter_pred),
                    original_probability=seed_prob,
                    counterfactual_probability=counter_prob,
                    changed_attributes=changed_attrs,
                    bias_type=f"{attr}_discrimination",
                    explanation=(
                        f"Changing {changed_attrs} changed "
                        f"prediction {seed_pred}→{counter_pred} "
                        f"(prob: {seed_prob:.3f}→{counter_prob:.3f})"
                    ),
                    metric_change=abs(metrics.spd),
                    attribute=attr
                )
                self.state.discriminatory_pairs.append(pair)
                self.state.good += 1
                return "G"
            else:
                self.state.failed += 1
                return "F"

        except Exception:
            self.state.useless += 1
            return "U"

    # ── MAIN AGENT LOOP ───────────────────────────────────────

    def run(self):
        """
        Main agentic loop:
        Perceive → Reason → Act → Evaluate → Reflect → Repeat
        """
        print("\n" + "=" * 65)
        print("  FairAgent: Bank Marketing ML Fairness Testing")
        print(f"  Dataset:   {len(self.loader.raw_df):,} customers")
        print(f"  Target:    {self.target_col}")
        print(f"  Protected: {self.protected_attrs}")
        print(f"  LLM:       {self.llm.model}")
        print("=" * 65)

        initial_metrics = None

        for iteration in range(self.n_iterations):
            print(f"\n{'─'*65}")
            print(
                f"  ITERATION {iteration+1}/{self.n_iterations}"
            )
            print(f"{'─'*65}")

            # ── PERCEIVE ───────────────────────────────────
            all_metrics = self._measure_fairness()
            if iteration == 0:
                initial_metrics = all_metrics

            biased_attrs = [
                attr for attr, m in all_metrics.items()
                if m.is_biased(self.fairness_threshold)
            ]

            # ── CHANGED: always search even if metrics look fair ──
            # Use all protected attrs if none detected as biased
            # Always search even if no bias above threshold
            if not biased_attrs:
                print("\n[INFO] No bias above threshold - searching all protected attrs")
                biased_attrs = self.protected_attrs
            if False:  # disabled early stop
                pass
            if not biased_attrs:
                print(
                    "\n[INFO] No bias detected above threshold "
                    "- searching all protected attrs anyway"
                )
                biased_attrs = self.protected_attrs
            if False:  # disabled original break
                print(
                    "\n[OK] No significant bias detected! "
                    "All metrics within threshold."
                )
                break

            print(f"\n[WARN]  Bias detected in: {biased_attrs}")

            # ── REASON: SHAP explanations ──────────────────
            print("\n[SEARCH] Computing feature importance (SHAP)...")
            shap_imp = self.shap.get_global_importance()
            for feat, val in list(shap_imp.items())[:5]:
                if isinstance(val, (list, np.ndarray)):
                   val = float(np.mean(val))
                print(f"  {feat:20s}: {val:.4f}")


            # ── ACT: Global Search ─────────────────────────
            primary_attr    = biased_attrs[0]
            primary_metrics = all_metrics[primary_attr]

            print(
                f"\n[GLOBAL] Global Search: generating "
                f"{self.n_seeds} seed customers..."
            )
            seeds = self.global_agent.generate_seeds(
                primary_metrics, shap_imp, self.n_seeds
            )
            print(f"  Generated {len(seeds)} seeds")

            # ── ACT: Local Search + EVALUATE + REFLECT ─────
            print(
                f"\n[LOCAL] Local Search "
                f"(G=found, F=not found, U=invalid, R=retry)"
            )

            for i, seed_raw in enumerate(seeds):
                print(
                    f"\r  [{i+1:2d}/{len(seeds)}] "
                    f"G={self.state.good} "
                    f"F={self.state.failed} "
                    f"U={self.state.useless} "
                    f"R={self.state.retries} "
                    f"cost=${self.llm.total_cost:.3f}",
                    end=""
                )

                for attr in biased_attrs:
                    metrics = all_metrics[attr]

                    # Encode seed for SHAP
                    seed_enc   = self.loader.encode_sample(
                        seed_raw
                    )
                    local_shap = self.shap.get_local_importance(
                        seed_enc
                    )

                    # Generate counterfactuals
                    cfs = self.local_agent.generate_counterfactuals(
                        seed_raw, local_shap,
                        metrics, self.n_counterfactuals
                    )

                    for cf_data in cfs:
                        result = self._evaluate_pair(
                            seed_raw, cf_data, attr, metrics
                        )

                        # ── REFLECT: Feedback loop ─────────
                        if result == "F":
                            for _ in range(self.max_retries):
                                self.state.retries += 1
                                new_cfs = (
                                    self.local_agent
                                    .retry_with_feedback(
                                        seed_raw, cf_data, metrics
                                    )
                                )
                                for new_cf in new_cfs:
                                    if isinstance(new_cf, dict):
                                        r = self._evaluate_pair(
                                            seed_raw, new_cf,
                                            attr, metrics
                                        )
                                        if r == "G":
                                            break

            print()
            self.state.total_cost = self.llm.total_cost
            print(
                f"\n  Iteration {iteration+1} summary:"
            )
            self.state.log()

        # ── FINAL RESULTS ──────────────────────────────────
        print("\n" + "=" * 65)
        print("  FINAL RESULTS")
        print("=" * 65)
        final_metrics = self._measure_fairness()
        print(
            f"\n  Discriminatory pairs found: "
            f"{self.state.good}"
        )
        print(
            f"  Total API cost: "
            f"${self.state.total_cost:.3f}"
        )

        # Save all outputs
        print("\n[FILES] Saving outputs...")
        self._save_discriminatory_pairs()
        self._save_report(initial_metrics, final_metrics)
        print("\n[CHARTS] Generating visualizations...")
        self.visualizer.plot_fairness_metrics(final_metrics)
        self.visualizer.plot_discrimination_heatmap(
            self.state.discriminatory_pairs
        )
        self.visualizer.plot_prediction_distribution(
            self.state.discriminatory_pairs
        )
        self.visualizer.plot_agent_progress(self.state)

        print("\n" + "=" * 65)
        print("  [OK] FairAgent completed!")
        print(f"  Results saved to: {self.output_dir}/")
        print("=" * 65)

        return self.state

    # ── SAVE OUTPUTS ──────────────────────────────────────────

    def _save_discriminatory_pairs(self):
        """
        Save discriminatory pairs in TWO formats:
        1. Full CSV with metadata
        2. Model-ready test dataset (just features + label)
           → can be loaded and run directly against any model
        """
        if not self.state.discriminatory_pairs:
            print("  No discriminatory pairs to save.")
            return

        rows = []
        for i, pair in enumerate(self.state.discriminatory_pairs):

            # Original customer row
            orig_row = {
                "pair_id":            i + 1,
                "patient_type":       "original",
                "bias_type":          pair.bias_type,
                "changed_attributes": str(pair.changed_attributes),
                "prediction":         pair.original_prediction,
                "probability":        round(
                    pair.original_probability, 4
                ),
                "explanation":        pair.explanation,
            }
            for feat, val in pair.original.items():
                orig_row[feat] = val
            orig_row[self.target_col] = pair.original_prediction
            rows.append(orig_row)

            # Counterfactual customer row
            cf_row = {
                "pair_id":            i + 1,
                "patient_type":       "counterfactual",
                "bias_type":          pair.bias_type,
                "changed_attributes": str(pair.changed_attributes),
                "prediction":         pair.counterfactual_prediction,
                "probability":        round(
                    pair.counterfactual_probability, 4
                ),
                "explanation":        pair.explanation,
            }
            for feat, val in pair.counterfactual.items():
                cf_row[feat] = val
            cf_row[self.target_col] = (
                pair.counterfactual_prediction
            )
            rows.append(cf_row)

        df = pd.DataFrame(rows)

        # 1. Full CSV with metadata
        full_path = os.path.join(
            self.output_dir, "discriminatory_pairs.csv"
        )
        df.to_csv(full_path, index=False)
        print(
            f"  [OK] discriminatory_pairs.csv "
            f"({len(self.state.discriminatory_pairs)} pairs, "
            f"{len(df)} rows)"
        )

        # 2. Model-ready test dataset
        # Remove metadata columns, keep only features + label
        meta_cols = [
            "pair_id", "patient_type", "bias_type",
            "changed_attributes", "prediction",
            "probability", "explanation"
        ]
        model_cols = [
            c for c in df.columns if c not in meta_cols
        ]
        model_df   = df[model_cols].copy()

        # Encode for model
        encoded_rows = []
        for _, row in model_df.iterrows():
            try:
                enc = self.loader.encode_sample(
                    row.to_dict()
                )
                encoded_rows.append(enc)
            except Exception:
                pass

        if encoded_rows:
            enc_df = pd.DataFrame(encoded_rows)
            model_path = os.path.join(
                self.output_dir,
                "test_dataset_model_ready.csv"
            )
            enc_df.to_csv(model_path, index=False)
            print(
                f"  [OK] test_dataset_model_ready.csv "
                f"(encoded, {len(enc_df)} rows)"
            )

        # 3. Usage instructions
        usage = f'''"""
How to use the model-ready test dataset
Generated by FairAgent
"""
import pandas as pd
from sklearn.metrics import classification_report

# ── Load the discriminatory test dataset ──────────────
test_df = pd.read_csv("test_dataset_model_ready.csv")

X_test = test_df.drop(columns=["{self.target_col}"])
y_true = test_df["{self.target_col}"]

# ── Run your model directly ────────────────────────────
# (replace 'model' with your trained model variable)
y_pred = model.predict(X_test)

print("Model performance on discriminatory cases:")
print(classification_report(y_true, y_pred))

# ── Compare original vs counterfactual ────────────────
pairs_df = pd.read_csv("discriminatory_pairs.csv")
orig   = pairs_df[pairs_df["patient_type"] == "original"]
counter= pairs_df[pairs_df["patient_type"] == "counterfactual"]

print(f"Original customers - positive rate: {{orig['prediction'].mean():.2%}}")
print(f"Counterfactual     - positive rate: {{counter['prediction'].mean():.2%}}")
print(f"Discrimination gap: {{abs(orig['prediction'].mean() - counter['prediction'].mean()):.2%}}")
'''

        usage_path = os.path.join(
            self.output_dir, "how_to_use_test_dataset.py"
        )
        with open(usage_path, 'w', encoding='utf-8') as f:
            f.write(usage)
        print(f"  [OK] how_to_use_test_dataset.py")

    def _save_report(
        self,
        initial_metrics: dict,
        final_metrics: dict
    ):
        """Save detailed fairness report"""
        path = os.path.join(
            self.output_dir, "fairness_report.txt"
        )
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 65 + "\n")
            f.write(
                "  FAIRAGENT - BANK MARKETING FAIRNESS REPORT\n"
            )
            f.write("=" * 65 + "\n\n")

            f.write("CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"  Dataset rows      : "
                f"{len(self.loader.raw_df):,}\n"
            )
            f.write(
                f"  Target column     : {self.target_col}\n"
            )
            f.write(
                f"  Protected attrs   : "
                f"{self.protected_attrs}\n"
            )
            f.write(
                f"  LLM model         : {self.llm.model}\n"
            )
            f.write(
                f"  Iterations        : {self.n_iterations}\n"
            )
            f.write(
                f"  Seeds per iter    : {self.n_seeds}\n"
            )
            f.write(
                f"  Counterfactuals   : "
                f"{self.n_counterfactuals}\n"
            )
            f.write(
                f"  Total API cost    : "
                f"${self.state.total_cost:.3f}\n\n"
            )

            f.write("FAIRNESS METRICS\n")
            f.write("-" * 40 + "\n")
            for attr in self.protected_attrs:
                m = final_metrics.get(attr)
                if not m:
                    continue
                f.write(f"\n  {attr.upper()}:\n")
                f.write(
                    f"    SPD = {m.spd:+.4f} "
                    f"{'[WARN] BIASED' if abs(m.spd) > self.fairness_threshold else '[OK] OK'}\n"
                )
                f.write(
                    f"    EOD = {m.eod:+.4f} "
                    f"{'[WARN] BIASED' if abs(m.eod) > self.fairness_threshold else '[OK] OK'}\n"
                )
                f.write(
                    f"    DI  = {m.di:.4f}  "
                    f"{'[WARN] BELOW 0.8 - LEGALLY PROBLEMATIC' if m.di < 0.8 else '[OK] OK'}\n"
                )
                f.write(
                    f"    AOD = {m.aod:+.4f} "
                    f"{'[WARN] BIASED' if abs(m.aod) > self.fairness_threshold else '[OK] OK'}\n"
                )
                f.write(
                    f"    Overall Bias Level: {m.bias_level()}\n"
                )

            f.write("\n\nTESTING RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"  G - Discriminatory pairs found : "
                f"{self.state.good}\n"
            )
            f.write(
                f"  F - No discrimination found    : "
                f"{self.state.failed}\n"
            )
            f.write(
                f"  U - Invalid samples            : "
                f"{self.state.useless}\n"
            )
            f.write(
                f"  R - Retries                    : "
                f"{self.state.retries}\n"
            )

            if self.state.discriminatory_pairs:
                f.write("\n\nBIAS BREAKDOWN\n")
                f.write("-" * 40 + "\n")
                bias_counts = {}
                for pair in self.state.discriminatory_pairs:
                    k = " + ".join(
                        sorted(pair.changed_attributes)
                    )
                    bias_counts[k] = (
                        bias_counts.get(k, 0) + 1
                    )
                for bt, cnt in sorted(
                    bias_counts.items(),
                    key=lambda x: x[1], reverse=True
                ):
                    f.write(f"  {bt}: {cnt} cases\n")

                f.write("\n\nSAMPLE CASES (first 5)\n")
                f.write("-" * 40 + "\n")
                for i, p in enumerate(
                    self.state.discriminatory_pairs[:5]
                ):
                    f.write(f"\n  Case {i+1}:\n")
                    f.write(
                        f"    Changed      : "
                        f"{p.changed_attributes}\n"
                    )
                    f.write(
                        f"    Original pred: "
                        f"{p.original_prediction} "
                        f"(prob={p.original_probability:.3f})\n"
                    )
                    f.write(
                        f"    Counter pred : "
                        f"{p.counterfactual_prediction} "
                        f"(prob="
                        f"{p.counterfactual_probability:.3f})\n"
                    )

            f.write("\n\nOUTPUT FILES\n")
            f.write("-" * 40 + "\n")
            f.write(
                "  discriminatory_pairs.csv      "
                "← full pairs with metadata\n"
                "  test_dataset_model_ready.csv  "
                "← run directly against model\n"
                "  how_to_use_test_dataset.py    "
                "← usage instructions\n"
                "  fairness_report.txt           "
                "← this report\n"
                "  visualizations/               "
                "← 4 charts\n"
            )

            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            for attr, m in final_metrics.items():
                if m.is_biased(self.fairness_threshold):
                    f.write(f"\n  {attr.upper()}:\n")
                    if abs(m.spd) > self.fairness_threshold:
                        f.write(
                            "  → Apply demographic parity "
                            "constraint\n"
                        )
                    if m.di < 0.8:
                        f.write(
                            "  → Disparate Impact below legal "
                            "threshold - urgent review needed\n"
                        )
                    if abs(m.eod) > self.fairness_threshold:
                        f.write(
                            "  → Apply equalized odds "
                            "post-processing\n"
                        )

        print(f"  [OK] fairness_report.txt")