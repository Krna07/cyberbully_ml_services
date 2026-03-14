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

# Leet-speak / obfuscation normalization map
LEET_MAP = {
    '@': 'a', '4': 'a', '3': 'e', '1': 'i', '!': 'i',
    '0': 'o', '5': 's', '$': 's', '7': 't', '+': 't',
    '8': 'b', '6': 'g',
}

def normalize_obfuscation(text):
    result = []
    for ch in text.lower():
        result.append(LEET_MAP.get(ch, ch))
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

# English model (primary)
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

# Hindi model
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

# common_model and roberta — commented out
# try:
#     common_model = joblib.load(model_path / "cyberbullyingcommmon_model.pkl")
# except Exception as e:
#     print(f"✗ common_model: {e}")

# try:
#     common_vectorizer = joblib.load(model_path / "tfidf_vectorizercommon.pkl")
# except Exception as e:
#     print(f"✗ common_vectorizer: {e}")

# roberta_classifier — commented out
# try:
#     from transformers import pipeline
#     roberta_classifier = pipeline(
#         "text-classification",
#         model="nayan90k/roberta-finetuned-cyberbullying-detection",
#         truncation=True, max_length=512
#     )
# except Exception as e:
#     print(f"✗ RoBERTa: {e}")

# ── Helpers ───────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

ENGLISH_TOXIC_KEYWORDS = [
    # Profanity
    'fuck', 'fucking', 'fucker', 'fucked', 'fck', 'fuk', 'wtf',
    'shit', 'shitty', 'bullshit', 'horseshit',
    'bitch', 'bitchy', 'bitches',
    'ass', 'asshole', 'arse', 'arsehole',
    'bastard', 'damn', 'crap', 'crappy',
    'dick', 'dickhead', 'cock', 'cockhead',
    'pussy', 'cunt', 'whore', 'slut', 'skank', 'hoe',
    'prick', 'twat', 'wanker', 'tosser', 'bellend',

    # Insults
    'stupid', 'stupido', 'idiot', 'idiotic',
    'hate', 'hater', 'hating',
    'kill', 'die', 'dead', 'death',
    'ugly', 'loser', 'dumb', 'dumbass', 'dumbhead',
    'worthless', 'pathetic', 'trash', 'garbage',
    'moron', 'fool', 'freak', 'freaky',
    'retard', 'retarded', 'imbecile', 'scum', 'scumbag',
    'pig', 'fat', 'fatso', 'fatty', 'lard',
    'disgusting', 'disgust', 'revolting',
    'smelly', 'stink', 'stinks', 'stinky',
    'gross', 'nasty', 'creep', 'creepy', 'weirdo', 'weird',
    'nobody', 'nothing', 'useless', 'hopeless', 'failure',
    'coward', 'wimp', 'crybaby', 'baby', 'clown',
    'joke', 'laughingstock', 'embarrassment',
    'shut up', 'shutup', 'go away', 'get lost', 'drop dead',
    'no one likes you', 'nobody likes you', 'everyone hates you',
    'go die', 'kys', 'kill yourself',

    # Homophobic / discriminatory
    'gay', 'fag', 'faggot', 'dyke', 'homo', 'queer',
    'tranny', 'transgender freak',
    'racist', 'sexist', 'bigot',

    # Threats
    'kill you', 'hurt you', 'beat you', 'destroy you', 'rape',
    'i will find you', 'watch your back', 'you will pay',
    'gonna get you', 'come after you',

    # Family insults
    'ur mum', 'your mum', 'ur mom', 'your mom',
    'ur dad', 'your dad', 'ur family', 'your family',
    'son of a bitch', 'son of a whore', 'motherf',

    # Body shaming
    'fat pig', 'ugly pig', 'cow', 'whale',
    'flat', 'tiny', 'small',

    # Misc
    'poop', 'poopie', 'poo', 'piss', 'pissed',
    'suck', 'sucks', 'sucker', 'lick my',
    'go to hell', 'burn in hell', 'rot in hell',
]

def extract_toxic_keywords(text, top_n=5):
    normalized = normalize_obfuscation(text)
    return [w for w in ENGLISH_TOXIC_KEYWORDS if w in normalized][:top_n]

def keyword_based_check(text):
    normalized = normalize_obfuscation(text)
    found = [w for w in ENGLISH_TOXIC_KEYWORDS if w in normalized]
    return len(found) > 0, found

def predict_english(text):
    """Primary English model prediction."""
    if not model or not vectorizer or not cleaner:
        return None
    normalized = normalize_obfuscation(text)
    cleaned = cleaner.transform([normalized])
    vec = vectorizer.transform(cleaned)
    pred = int(model.predict(vec)[0])
    conf = float(max(model.predict_proba(vec)[0]))
    return {"prediction": pred, "confidence": round(conf, 2), "model_used": "cyberbullying_model"}

def build_categories(prediction, text):
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
                return {
                    "prediction": "Cyberbullying Detected" if pred == 1 else "Safe Message",
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

        # ── English ──
        kw_toxic, kw_found = keyword_based_check(input_data.text)
        eng = predict_english(input_data.text)

        if eng is not None:
            pred = eng["prediction"]
            confidence = eng["confidence"]
            model_used = eng["model_used"]
            # keyword override — explicit slurs override model
            if kw_toxic and pred == 0:
                pred = 1
                confidence = max(confidence, 0.80)
                model_used += "+keyword_override"
        else:
            # no model loaded — keyword only
            pred = 1 if kw_toxic else 0
            confidence = 0.80 if kw_toxic else 0.70
            model_used = "keyword_fallback"

        return {
            "prediction": "Cyberbullying Detected" if pred == 1 else "Safe Message",
            "confidence": confidence,
            "categories": build_categories(pred, input_data.text),
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
            "english_model": model is not None,
            "english_vectorizer": vectorizer is not None,
            "text_cleaner": cleaner is not None,
            "hindi_model": hindi_model is not None,
            "hindi_vectorizer": hindi_vectorizer is not None,
        }
    }

@app.get("/")
async def root():
    return {"message": "Cyberbullying Detection ML Service", "version": "2.0.0"}
