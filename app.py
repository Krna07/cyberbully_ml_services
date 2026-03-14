from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import joblib
import numpy as np
from pathlib import Path
import re
from multilingual_data import HINDI_TOXIC_WORDS, TELUGU_TOXIC_WORDS

# RoBERTa model (loaded lazily to avoid blocking startup)
roberta_classifier = None
try:
    from transformers import pipeline
    roberta_classifier = pipeline(
        "text-classification",
        model="nayan90k/roberta-finetuned-cyberbullying-detection",
        truncation=True,
        max_length=512
    )
    print("✓ RoBERTa cyberbullying model loaded")
except Exception as e:
    print(f"✗ RoBERTa model not loaded: {e}")

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

# Leet-speak / obfuscation normalization map
LEET_MAP = {
    '@': 'a', '4': 'a', '3': 'e', '1': 'i', '!': 'i',
    '0': 'o', '5': 's', '$': 's', '7': 't', '+': 't',
    '8': 'b', '6': 'g',
}

def normalize_obfuscation(text):
    """Expand leet-speak and common obfuscation tricks before model inference."""
    result = []
    for ch in text.lower():
        result.append(LEET_MAP.get(ch, ch))
    text = ''.join(result)
    # f**k / f*ck / f-u-c-k style
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

# ── Load models ──────────────────────────────────────────────────────────────
model_path = Path(__file__).parent
model = vectorizer = cleaner = None
hindi_model = hindi_vectorizer = None
common_model = common_vectorizer = None

try:
    with open(model_path / "cyberbullying_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✓ cyberbullying_model loaded")
except Exception as e:
    print(f"✗ cyberbullying_model: {e}")

try:
    with open(model_path / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✓ tfidf_vectorizer loaded")
except Exception as e:
    print(f"✗ tfidf_vectorizer: {e}")

try:
    with open(model_path / "text_cleaner.pkl", "rb") as f:
        cleaner = pickle.load(f)
    print("✓ text_cleaner loaded")
except Exception as e:
    print(f"✗ text_cleaner: {e} — using fallback")
    cleaner = SimpleTextCleaner()

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

try:
    common_model = joblib.load(model_path / "cyberbullyingcommmon_model.pkl")
    print("✓ common_model loaded (LogisticRegression)")
except Exception as e:
    print(f"✗ common_model: {e}")

try:
    common_vectorizer = joblib.load(model_path / "tfidf_vectorizercommon.pkl")
    print(f"✓ common_vectorizer loaded (vocab={len(common_vectorizer.get_feature_names_out())})")
except Exception as e:
    print(f"✗ common_vectorizer: {e}")

# ── Helpers ───────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

ENGLISH_TOXIC_KEYWORDS = [
    # Profanity
    'fuck', 'fucking', 'fucker', 'fck', 'fuk',
    'shit', 'bitch', 'ass', 'asshole', 'bastard', 'damn', 'crap',
    'dick', 'cock', 'pussy', 'cunt', 'whore', 'slut',
    # Insults
    'stupid', 'idiot', 'hate', 'kill', 'die', 'ugly', 'loser', 'dumb',
    'worthless', 'pathetic', 'trash', 'garbage', 'moron', 'fool', 'freak',
    'retard', 'retarded', 'imbecile', 'scum', 'pig', 'fat', 'disgusting',
    'smelly', 'stink', 'stinks', 'gross', 'nasty', 'creep', 'weirdo',
    # Homophobic / discriminatory
    'gay', 'fag', 'faggot', 'dyke', 'homo', 'queer',
    # Threats
    'kill you', 'hurt you', 'beat you', 'destroy you', 'rape',
    # Mum/family insults
    'ur mum', 'your mum', 'ur mom', 'your mom', 'mother',
    # Misc
    'poop', 'poopie', 'poo', 'crap',
]

def extract_toxic_keywords(text, top_n=5):
    normalized = normalize_obfuscation(text)
    return [w for w in ENGLISH_TOXIC_KEYWORDS if w in normalized][:top_n]

def keyword_based_check(text):
    """Fast keyword pre-check on normalized text. Returns (is_toxic, found_words)."""
    normalized = normalize_obfuscation(text)
    found = [w for w in ENGLISH_TOXIC_KEYWORDS if w in normalized]
    return len(found) > 0, found

def ensemble_predict_english(text):
    """Run both English models and combine via soft voting."""
    normalized = normalize_obfuscation(text)
    cleaned = cleaner.transform([normalized])
    results = []

    if model and vectorizer:
        vec1 = vectorizer.transform(cleaned)
        pred1 = int(model.predict(vec1)[0])
        conf1 = float(max(model.predict_proba(vec1)[0]))
        results.append((pred1, conf1))

    if common_model and common_vectorizer:
        text_clean = re.sub(r'[^a-z0-9\s]', '', normalized)
        text_clean = ' '.join(text_clean.split())
        vec2 = common_vectorizer.transform([text_clean])
        pred2 = int(common_model.predict(vec2)[0])
        conf2 = float(max(common_model.predict_proba(vec2)[0]))
        results.append((pred2, conf2))

    if not results:
        return None

    # If either model flags bullying → bullying; average confidence
    final_pred = 1 if any(r[0] == 1 for r in results) else 0
    final_conf = round(sum(r[1] for r in results) / len(results), 2)

    models_used = []
    if model and vectorizer:
        models_used.append("cyberbullying_model")
    if common_model and common_vectorizer:
        models_used.append("common_model")

    return {
        "prediction": final_pred,
        "confidence": final_conf,
        "models_used": "+".join(models_used),
    }

def roberta_predict(text):
    """Run RoBERTa model. Returns (is_toxic: bool, confidence: float) or None if unavailable."""
    if not roberta_classifier:
        return None, None
    try:
        result = roberta_classifier(text[:512])[0]
        label = result["label"].lower()
        score = float(result["score"])
        # label is typically 'cyberbullying' or 'not cyberbullying' / 'LABEL_1' / 'LABEL_0'
        is_toxic = "bully" in label or label == "label_1" or label == "1"
        return is_toxic, round(score, 2)
    except Exception as e:
        print(f"RoBERTa predict error: {e}")
        return None, None


    text_lower = text.lower()
    cats = {k: 0 for k in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]}
    if prediction == 1:
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
                result = "Cyberbullying Detected" if pred == 1 else "Safe Message"
                return {
                    "prediction": result,
                    "confidence": round(conf, 2),
                    "categories": build_categories(pred, input_data.text),
                    "toxicKeywords": extract_toxic_keywords(input_data.text) if pred == 1 else [],
                    "language": language,
                    "model_used": "hindi_trained_model"
                }
            except Exception as e:
                print(f"Hindi model error: {e}, falling back to keyword matching")

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

        # ── English ensemble ──
        if not cleaner:
            raise HTTPException(status_code=500, detail="Text cleaner not loaded")

        # Keyword pre-check catches obfuscated/slang text the models may miss
        kw_toxic, kw_found = keyword_based_check(input_data.text)

        ensemble = ensemble_predict_english(input_data.text)

        # RoBERTa — final and most accurate layer
        roberta_toxic, roberta_conf = roberta_predict(input_data.text)

        # Decision logic:
        # 1. RoBERTa available → trust it as primary
        # 2. Keyword override if RoBERTa says safe but explicit keywords found
        # 3. Fall back to ensemble if RoBERTa unavailable
        if roberta_toxic is not None:
            pred = 1 if roberta_toxic else 0
            confidence = roberta_conf
            model_used = "roberta"
            # keyword override even on roberta — explicit slurs are unambiguous
            if kw_toxic and pred == 0:
                pred = 1
                confidence = max(confidence, 0.80)
                model_used = "roberta+keyword_override"
        elif ensemble is not None:
            pred = ensemble["prediction"]
            confidence = ensemble["confidence"]
            model_used = ensemble["models_used"]
            if kw_toxic and pred == 0:
                pred = 1
                confidence = max(confidence, 0.78)
                model_used += "+keyword_override"
        else:
            pred = 1 if kw_toxic else 0
            confidence = 0.80 if kw_toxic else 0.70
            model_used = "keyword_fallback"

        result = "Cyberbullying Detected" if pred == 1 else "Safe Message"
        response = {
            "prediction": result,
            "confidence": confidence,
            "categories": build_categories(pred, input_data.text),
            "toxicKeywords": kw_found if pred == 1 else [],
            "language": "english",
            "model_used": model_used
        }
        return response

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
            "english_model": model is not None,
            "english_vectorizer": vectorizer is not None,
            "text_cleaner": cleaner is not None,
            "hindi_model": hindi_model is not None,
            "hindi_vectorizer": hindi_vectorizer is not None,
            "common_model": common_model is not None,
            "common_vectorizer": common_vectorizer is not None,
            "roberta_model": roberta_classifier is not None,
        }
    }

@app.get("/")
async def root():
    return {"message": "Cyberbullying Detection ML Service", "version": "2.0.0"}
