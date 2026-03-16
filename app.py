from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from pathlib import Path
import re
from multilingual_data import HINDI_TOXIC_WORDS, TELUGU_TOXIC_WORDS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleTextCleaner:
    def transform(self, texts):
        cleaned = []
        for text in texts:
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', '', text)
            text = ' '.join(text.split())
            cleaned.append(text)
        return cleaned

LEET_MAP = {
    '@': 'a', '4': 'a', '3': 'e', '1': 'i', '!': 'i',
    '0': 'o', '5': 's', '$': 's', '7': 't', '+': 't',
    '8': 'b', '6': 'g',
}

def normalize_obfuscation(text):
    result = [LEET_MAP.get(ch, ch) for ch in text.lower()]
    text = ''.join(result)
    text = re.sub(r'f[\*\-_\.]+u?[\*\-_\.]*c?[\*\-_\.]*k', 'fuck', text)
    text = re.sub(r's[\*\-_\.]+h[\*\-_\.]*i[\*\-_\.]*t', 'shit', text)
    text = re.sub(r'b[\*\-_\.]+i[\*\-_\.]*t[\*\-_\.]*c[\*\-_\.]*h', 'bitch', text)
    text = re.sub(r'a[\*\-_\.]+s[\*\-_\.]*s', 'ass', text)
    return text

def detect_language(text):
    if re.search(r'[\u0900-\u097F]', text):
        return 'hindi'
    if re.search(r'[\u0C00-\u0C7F]', text):
        return 'telugu'
    text_lower = text.lower()
    if sum(1 for word in HINDI_TOXIC_WORDS if word in text_lower) > 0:
        return 'hindi'
    if sum(1 for word in TELUGU_TOXIC_WORDS if word in text_lower) > 0:
        return 'telugu'
    return 'english'

def check_multilingual_toxicity(text, language):
    text_lower = text.lower()
    toxic_words = []
    if language == 'hindi':
        toxic_words = [w for w in HINDI_TOXIC_WORDS if w in text_lower]
    elif language == 'telugu':
        toxic_words = [w for w in TELUGU_TOXIC_WORDS if w in text_lower]
    is_toxic = len(toxic_words) > 0
    confidence = min(0.95, 0.6 + (len(toxic_words) * 0.15))
    return is_toxic, toxic_words, confidence

# ── Load models ───────────────────────────────────────────────────────────────
model_path = Path(__file__).parent
model = vectorizer = cleaner = None
hindi_model = hindi_vectorizer = None
distilbert_model = distilbert_tokenizer = None

# ── DistilBERT (primary English — loads if available) ─────────────────────
try:
    import torch
    with open(model_path / "toxic_bullying_model.pkl", "rb") as f:
        distilbert_model = pickle.load(f)
    with open(model_path / "tokenizer.pkl", "rb") as f:
        distilbert_tokenizer = pickle.load(f)
    distilbert_model.eval()
    print("✓ DistilBERT toxic_bullying_model loaded")
except Exception as e:
    print(f"✗ DistilBERT not loaded: {e} — falling back to sklearn")

# ── sklearn fallback English model ───────────────────────────────────────
if not distilbert_model:
    try:
        with open(model_path / "cyberbullying_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(model_path / "tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        print("✓ sklearn cyberbullying_model loaded (fallback)")
    except Exception as e:
        print(f"✗ sklearn model: {e}")

try:
    with open(model_path / "text_cleaner.pkl", "rb") as f:
        cleaner = pickle.load(f)
    print("✓ text_cleaner loaded")
except Exception as e:
    print(f"✗ text_cleaner: {e} — using fallback")
    cleaner = SimpleTextCleaner()

# ── Hindi model ───────────────────────────────────────────────────────────
try:
    with open(model_path / "hindi_cyberbullying_model_v2.pkl", "rb") as f:
        hindi_model = pickle.load(f)
    print("✓ hindi_model loaded")
except Exception as e:
    print(f"✗ hindi_model: {e}")

try:
    with open(model_path / "hindi_vectorizer_v2.pkl", "rb") as f:
        hindi_vectorizer = pickle.load(f)
    print("✓ hindi_vectorizer loaded")
except Exception as e:
    print(f"✗ hindi_vectorizer: {e}")

# ── Helpers ───────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

DISTILBERT_LABEL_MAP = {
    0: 'toxic', 1: 'severe_toxic', 2: 'obscene',
    3: 'threat', 4: 'insult', 5: 'identity_hate'
}

ENGLISH_TOXIC_KEYWORDS = [
    'fuck', 'fucking', 'fucker', 'fucked', 'fck', 'fuk', 'wtf',
    'shit', 'shitty', 'bullshit', 'bitch', 'bitchy', 'bitches',
    'ass', 'asshole', 'arse', 'arsehole', 'bastard', 'damn', 'crap',
    'dick', 'dickhead', 'cock', 'pussy', 'cunt', 'whore', 'slut',
    'skank', 'hoe', 'prick', 'twat', 'wanker', 'tosser', 'bellend',
    'stupid', 'idiot', 'hate', 'kill', 'die', 'ugly', 'loser', 'dumb',
    'dumbass', 'worthless', 'pathetic', 'trash', 'garbage', 'moron',
    'fool', 'freak', 'retard', 'retarded', 'imbecile', 'scum', 'scumbag',
    'pig', 'fat', 'fatso', 'fatty', 'disgusting', 'smelly', 'stink',
    'stinks', 'stinky', 'gross', 'nasty', 'creep', 'weirdo',
    'nobody', 'useless', 'hopeless', 'failure', 'coward', 'wimp',
    'crybaby', 'clown', 'shut up', 'shutup', 'go away', 'get lost',
    'drop dead', 'no one likes you', 'nobody likes you', 'go die',
    'kys', 'kill yourself', 'everyone hates you',
    'gay', 'fag', 'faggot', 'dyke', 'homo', 'queer',
    'kill you', 'hurt you', 'beat you', 'destroy you', 'rape',
    'ur mum', 'your mum', 'ur mom', 'your mom',
    'son of a bitch', 'motherf',
    'poop', 'poopie', 'poo', 'piss',
    'suck', 'sucks', 'go to hell', 'burn in hell',
]

def extract_toxic_keywords(text, top_n=5):
    normalized = normalize_obfuscation(text)
    return [w for w in ENGLISH_TOXIC_KEYWORDS if w in normalized][:top_n]

def keyword_based_check(text):
    normalized = normalize_obfuscation(text)
    found = [w for w in ENGLISH_TOXIC_KEYWORDS if w in normalized]
    return len(found) > 0, found

def predict_english_distilbert(text):
    try:
        import torch, torch.nn as nn
        inputs = distilbert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            logits = distilbert_model(**inputs).logits
        probs = nn.Sigmoid()(logits)[0]
        detected = {DISTILBERT_LABEL_MAP[i]: round(float(probs[i]), 2) for i in range(6) if probs[i] > 0.5}
        is_toxic = len(detected) > 0
        return {
            "prediction": 1 if is_toxic else 0,
            "confidence": round(float(max(probs)), 2),
            "model_used": "distilbert_toxic_bullying",
            "detected_labels": detected
        }
    except Exception as e:
        print(f"DistilBERT predict error: {e}")
        return None

def predict_english_sklearn(text):
    if not model or not vectorizer or not cleaner:
        return None
    normalized = normalize_obfuscation(text)
    cleaned = cleaner.transform([normalized])
    vec = vectorizer.transform(cleaned)
    pred = int(model.predict(vec)[0])
    conf = float(max(model.predict_proba(vec)[0]))
    return {"prediction": pred, "confidence": round(conf, 2), "model_used": "cyberbullying_model", "detected_labels": {}}

def build_categories(prediction, text, detected_labels=None):
    cats = {k: 0 for k in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]}
    if prediction == 1:
        if detected_labels:
            for label in detected_labels:
                if label in cats:
                    cats[label] = 1
        else:
            text_lower = text.lower()
            cats["toxic"] = 1
            if any(w in text_lower for w in ['kill', 'die', 'death', 'madarchod', 'behenchod']):
                cats["severe_toxic"] = 1
            if any(w in text_lower for w in ['damn', 'hell', 'crap', 'fuck']):
                cats["obscene"] = 1
            if any(w in text_lower for w in ['kill you', 'hurt you', 'destroy you', 'maar', 'marunga']):
                cats["threat"] = 1
            if any(w in text_lower for w in ['stupid', 'idiot', 'dumb', 'moron', 'fool', 'loser', 'ugly', 'chutiya', 'bewakoof']):
                cats["insult"] = 1
            if any(w in text_lower for w in ['hate', 'racist', 'sexist']):
                cats["identity_hate"] = 1
    return cats

# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(input_data: TextInput):
    try:
        language = detect_language(input_data.text)

        # ── Hindi ──
        if language == 'hindi' and hindi_model and hindi_vectorizer:
            try:
                cleaned = cleaner.transform([input_data.text])
                vec = hindi_vectorizer.transform(cleaned)
                pred = int(hindi_model.predict(vec)[0])
                conf = float(max(hindi_model.predict_proba(vec)[0]))
                return {
                    "prediction": "Cyberbullying Detected" if pred == 1 else "Safe Message",
                    "confidence": round(conf, 2),
                    "categories": build_categories(pred, input_data.text),
                    "toxicKeywords": extract_toxic_keywords(input_data.text) if pred == 1 else [],
                    "language": language,
                    "model_used": "hindi_trained_model"
                }
            except Exception as e:
                print(f"Hindi model error: {e}")

        # ── Hindi / Telugu keyword fallback ──
        if language in ['hindi', 'telugu']:
            is_toxic, toxic_words, confidence = check_multilingual_toxicity(input_data.text, language)
            pred = 1 if is_toxic else 0
            return {
                "prediction": "Cyberbullying Detected" if is_toxic else "Safe Message",
                "confidence": round(confidence, 2),
                "categories": build_categories(pred, input_data.text),
                "toxicKeywords": toxic_words,
                "language": language,
                "model_used": "keyword_matching"
            }

        # ── English ──
        kw_toxic, kw_found = keyword_based_check(input_data.text)

        # Try DistilBERT first, fall back to sklearn
        eng = predict_english_distilbert(input_data.text) if distilbert_model else None
        if eng is None:
            eng = predict_english_sklearn(input_data.text)

        if eng is not None:
            pred = eng["prediction"]
            confidence = eng["confidence"]
            model_used = eng["model_used"]
            detected_labels = eng.get("detected_labels", {})
            if kw_toxic and pred == 0:
                pred = 1
                confidence = max(confidence, 0.80)
                model_used += "+keyword_override"
                detected_labels = {}
        else:
            pred = 1 if kw_toxic else 0
            confidence = 0.80 if kw_toxic else 0.70
            model_used = "keyword_fallback"
            detected_labels = {}

        return {
            "prediction": "Cyberbullying Detected" if pred == 1 else "Safe Message",
            "confidence": confidence,
            "categories": build_categories(pred, input_data.text, detected_labels if pred == 1 else None),
            "toxicKeywords": kw_found if pred == 1 else [],
            "language": "english",
            "model_used": model_used
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": {
            "distilbert_english": distilbert_model is not None,
            "distilbert_tokenizer": distilbert_tokenizer is not None,
            "sklearn_fallback": model is not None,
            "hindi_model": hindi_model is not None,
            "hindi_vectorizer": hindi_vectorizer is not None,
        }
    }

@app.get("/")
async def root():
    return {"message": "Cyberbullying Detection ML Service", "version": "3.0.0"}
