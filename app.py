# app.py
import streamlit as st
import numpy as np
import pandas as pd
import math
import re
from typing import Dict, Tuple, List, Optional

# NLP libs
import transformers
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import torch
from nltk.corpus import stopwords as nltk_stopwords
import nltk

# ---------------------------
# Streamlit setup
# ---------------------------
st.set_page_config(page_title="AI / Human Text Detector", layout="wide")
st.title("🕵️ AI / Human Text Detector")
st.markdown("""
輸入文字後立即判斷 **AI% / Human%**，可選擇方法：
- **Heuristic**（自建特徵，快速）
- **Sklearn**（TF-IDF + Logistic）
- **Transformer**（Hugging Face 預訓練模型）
""")

# ---------------------------
# Utilities
# ---------------------------
PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)

def basic_text_clean(text: str) -> str:
    return text.strip()

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def avg_word_length(text: str) -> float:
    words = tokenize_words(text)
    return np.mean([len(w) for w in words]) if words else 0.0

def stopword_ratio(text: str, stopwords:set) -> float:
    words = tokenize_words(text)
    if not words: return 0.0
    return sum(1 for w in words if w in stopwords) / len(words)

def punctuation_ratio(text: str) -> float:
    punct = len(PUNCT_RE.findall(text))
    total = len(text) if text else 1
    return punct / total

def type_token_ratio(text: str) -> float:
    words = tokenize_words(text)
    if not words: return 0.0
    return len(set(words)) / len(words)

def sentence_stats(text: str) -> Tuple[float, float]:
    sents = re.split(r'[.!?]+\s*', text.strip())
    sents = [s for s in sents if s.strip()]
    if not sents:
        return 0.0, 0.0
    lens = [len(tokenize_words(s)) for s in sents]
    return float(np.mean(lens)), float(len(sents))

def normalize_feature(value, minv, maxv):
    if maxv - minv == 0:
        return 0.5
    return (value - minv) / (maxv - minv)

def compute_features(text: str, stopwords:set) -> Dict[str, float]:
    txt = basic_text_clean(text)
    awl = avg_word_length(txt)
    swr = stopword_ratio(txt, stopwords)
    pr = punctuation_ratio(txt)
    ttr = type_token_ratio(txt)
    asl, sents = sentence_stats(txt)
    feats = {
        "avg_word_len": awl,
        "stopword_ratio": swr,
        "punct_ratio": pr,
        "type_token_ratio": ttr,
        "avg_sent_len": asl
    }
    return feats

def heuristic_score(features: Dict[str,float], weights: Dict[str,float]) -> float:
    ranges = {
        "avg_word_len": (2.0, 8.0),
        "stopword_ratio": (0.0, 1.0),
        "punct_ratio": (0.0, 0.5),
        "type_token_ratio": (0.05, 1.0),
        "avg_sent_len": (2.0, 80.0)
    }
    total = 0.0
    weight_sum = 0.0
    for k,w in weights.items():
        v = features.get(k,0.0)
        minv, maxv = ranges.get(k,(0.0,1.0))
        nv = normalize_feature(v,minv,maxv)
        total += nv*w
        weight_sum += abs(w)
    if weight_sum==0:
        return 0.5
    score = 1 / (1 + math.exp(- (total / weight_sum)*3 ))  # logistic
    return float(score)

def train_tfidf_logistic(texts: List[str], labels: List[int]):
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    pipe = make_pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1,2)), LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:,1]
    report = classification_report(y_test, preds, output_dict=True)
    auc = None
    try:
        auc = roc_auc_score(y_test, proba)
    except:
        auc = None
    return pipe, report, auc

# ---------------------------
# Load stopwords
# ---------------------------
try:
    _ = nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
DEFAULT_STOPWORDS = set(nltk_stopwords.words('english'))

# ---------------------------
# Sidebar UI
# ---------------------------
col1, col2 = st.columns([2,1])
with col1:
    user_text = st.text_area("請輸入要判斷的文本：", height=200)
    method = st.selectbox("選擇判斷方法：", ["Heuristic", "Sklearn", "Transformer"])
    run_button = st.button("判斷 (Run)")
with col2:
    show_visuals = st.checkbox("顯示特徵統計圖", value=True)
    uploaded_csv = st.file_uploader("上傳 training CSV (text,label 1=AI,0=Human)", type=["csv"])
    hf_model_name = st.text_input("Transformer model (HF) 名稱，例如 roberta-base", value="")

# ---------------------------
# Main execution
# ---------------------------
output_placeholder = st.empty()

if run_button:
    if not user_text.strip():
        st.warning("請先輸入文本")
    else:
        feats = compute_features(user_text, DEFAULT_STOPWORDS)
        if method=="Heuristic":
            weights = {
                "avg_word_len": -0.5,
                "stopword_ratio": -0.5,
                "punct_ratio": -0.3,
                "type_token_ratio": 0.7,
                "avg_sent_len": -0.3
            }
            score = heuristic_score(feats, weights)
            ai_pct = round(score*100,1)
            human_pct = round(100-ai_pct,1)
            output_placeholder.markdown(f"## 判斷結果：AI {ai_pct}% | Human {human_pct}%")
            if show_visuals:
                df_feat = pd.DataFrame.from_dict(feats, orient='index', columns=['value'])
                fig,ax = plt.subplots(figsize=(6,3))
                df_feat['value'].plot(kind='bar',ax=ax)
                ax.set_ylabel("Value")
                ax.set_title("特徵統計")
                st.pyplot(fig)

        elif method=="Sklearn":
            # 如果沒上傳 CSV，使用內建範例
            if uploaded_csv is None:
                st.info("未上傳 CSV，使用內建範例訓練 Sklearn 模型")
                # 簡單內建訓練資料
                texts = [
                    "Artificial intelligence is transforming the world in unprecedented ways.", # AI
                    "AI models can generate text, images, and even code automatically.", # AI
                    "I went to the supermarket to buy some fruits today.", # Human
                    "Yesterday I went hiking with my friends and it was amazing.", # Human
                    "The stock market has been volatile due to recent events.", # Human
                    "GPT models can assist in writing articles and emails quickly.", # AI
                    "I just finished my homework and now I'm watching TV.", # Human
                    "Machine learning algorithms can learn patterns from large datasets.", # AI
                ]
                labels = [1,1,0,0,0,1,0,1]
            else:
                df = pd.read_csv(uploaded_csv)
                if 'text' not in df.columns or 'label' not in df.columns:
                    st.error("CSV 必須包含 text 與 label 欄位")
                else:
                    texts = df['text'].astype(str).tolist()
                    labels = df['label'].astype(int).tolist()
            model_pipe, report, auc = train_tfidf_logistic(texts, labels)
            proba = model_pipe.predict_proba([user_text])[0,1]
            ai_pct = round(proba*100,1)
            human_pct = round(100-ai_pct,1)
            output_placeholder.markdown(f"## 判斷結果（Sklearn）：AI {ai_pct}% | Human {human_pct}%")
            if show_visuals:
                vec = model_pipe.named_steps['tfidfvectorizer']
                clf = model_pipe.named_steps['logisticregression']
                try:
                    feat_names = vec.get_feature_names_out()
                    coefs = clf.coef_[0]
                    top_idx = np.argsort(np.abs(coefs))[-20:]
                    top_df = pd.DataFrame({
                        'feature':[feat_names[i] for i in top_idx],
                        'coef':[coefs[i] for i in top_idx]
                    })
                    fig,ax = plt.subplots(figsize=(6,4))
                    top_df.set_index('feature')['coef'].plot(kind='bar',ax=ax)
                    ax.set_title("Top 20 特徵係數")
                    st.pyplot(fig)
                except:
                    st.info("無法顯示特徵重要性")

        elif method=="Transformer":
            if not hf_model_name:
                st.warning("請輸入 Hugging Face 模型名稱")
            else:
                try:
                    classifier = pipeline("text-classification", model=hf_model_name, tokenizer=hf_model_name, return_all_scores=True)
                    preds = classifier(user_text[:10000])
                    if isinstance(preds[0], list):
                        cls_probs = preds[0]
                    else:
                        cls_probs = preds
                    # 嘗試找出 AI 標籤
                    ai_score = None
                    for p in cls_probs:
                        if 'ai' in p['label'].lower() or 'machine' in p['label'].lower() or 'generated' in p['label'].lower():
                            ai_score = p['score']
                            break
                    if ai_score is None:
                        # fallback 顯示原始標籤
                        st.json({p['label']:p['score'] for p in cls_probs})
                        st.info("無法判斷哪個標籤代表 AI")
                    else:
                        ai_pct = round(ai_score*100,1)
                        human_pct = round(100-ai_pct,1)
                        output_placeholder.markdown(f"## 判斷結果（Transformer）：AI {ai_pct}% | Human {human_pct}%")
                except Exception as e:
                    st.error(f"Transformer 推論失敗: {e}")
