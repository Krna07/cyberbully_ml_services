from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from pathlib import Path
import re
from multilingual_data import HINDI_TOXIC_WORDS, TELUGU_TOXIC_WORDS

app = FastAPI()

# CORS configuration for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://cyberbully-backend.onrender.com",
        "*"  # Allow all origins for now
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple text cleaner class as fallback
class SimpleTextCleaner:
    def transform(self, texts):
        cleaned = []
        for text in texts:
            # Convert to lowercase
            text = text.lower()
            # Remove special characters but keep spaces
            text = re.sub(r'[^a-z0-9\s]', '', text)
            # Remove extra spaces
            text = ' '.join(text.split())
            cleaned.append(text)
        return cleaned

def detect_language(text):
    """Detect if text is Hindi, Telugu, or English"""
    # Check for Devanagari script (Hindi)
    if re.search(r'[\u0900-\u097F]', text):
        return 'hindi'
    # Check for Telugu script
    if re.search(r'[\u0C00-\u0C7F]', text):
        return 'telugu'
    # Check for Hindi/Telugu words in Latin script
    text_lower = text.lower()
    hindi_matches = sum(1 for word in HINDI_TOXIC_WORDS if word in text_lower)
    telugu_matches = sum(1 for word in TELUGU_TOXIC_WORDS if word in text_lower)
    
    if hindi_matches > 0:
        return 'hindi'
    if telugu_matches > 0:
        return 'telugu'
    
    return 'english'

def check_multilingual_toxicity(text, language):
    """Check for toxic words in Hindi/Telugu"""
    text_lower = text.lower()
    toxic_words = []
    
    if language == 'hindi':
        toxic_words = [word for word in HINDI_TOXIC_WORDS if word in text_lower]
    elif language == 'telugu':
        toxic_words = [word for word in TELUGU_TOXIC_WORDS if word in text_lower]
    
    is_toxic = len(toxic_words) > 0
    confidence = min(0.95, 0.6 + (len(toxic_words) * 0.15))
    
    return is_toxic, toxic_words, confidence

# Load models
model_path = Path(__file__).parent
model = vectorizer = cleaner = None
hindi_model = hindi_vectorizer = None

try:
    print("Loading cyberbullying_model.pkl...")
    with open(model_path / "cyberbullying_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")

try:
    print("Loading tfidf_vectorizer.pkl...")
    with open(model_path / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✓ Vectorizer loaded successfully")
except Exception as e:
    print(f"✗ Error loading vectorizer: {e}")

try:
    print("Loading text_cleaner.pkl...")
    with open(model_path / "text_cleaner.pkl", "rb") as f:
        cleaner = pickle.load(f)
    print("✓ Text cleaner loaded successfully")
except Exception as e:
    print(f"✗ Error loading text cleaner: {e}")
    print("Using fallback SimpleTextCleaner")
    cleaner = SimpleTextCleaner()

# Load Hindi models
try:
    print("Loading hindi_cyberbullying_model_v2.pkl...")
    with open(model_path / "hindi_cyberbullying_model_v2.pkl", "rb") as f:
        hindi_model = pickle.load(f)
    print("✓ Hindi model loaded successfully")
except Exception as e:
    print(f"✗ Error loading Hindi model: {e}")

try:
    print("Loading hindi_vectorizer_v2.pkl...")
    with open(model_path / "hindi_vectorizer_v2.pkl", "rb") as f:
        hindi_vectorizer = pickle.load(f)
    print("✓ Hindi vectorizer loaded successfully")
except Exception as e:
    print(f"✗ Error loading Hindi vectorizer: {e}")

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    toxicKeywords: list = []
    categories: dict = {}

def extract_toxic_keywords(text, vectorized, feature_names, top_n=5):
    """Extract potential toxic keywords from the text"""
    toxic_words = [
        'stupid', 'idiot', 'hate', 'kill', 'die', 'ugly', 'loser', 'dumb',
        'worthless', 'pathetic', 'trash', 'garbage', 'moron', 'fool', 'freak'
    ]
    
    found_keywords = []
    text_lower = text.lower()
    
    for word in toxic_words:
        if word in text_lower:
            found_keywords.append(word)
    
    return found_keywords[:top_n]

@app.post("/predict")
async def predict(input_data: TextInput):
    try:
        # Detect language
        language = detect_language(input_data.text)
        
        # For Hindi, use trained Hindi model if available
        if language == 'hindi' and hindi_model and hindi_vectorizer:
            try:
                print(f"Using trained Hindi model for: {input_data.text[:50]}")
                
                # Clean and vectorize Hindi text
                cleaned_text = cleaner.transform([input_data.text])
                vectorized = hindi_vectorizer.transform(cleaned_text)
                
                # Predict using Hindi model
                prediction = hindi_model.predict(vectorized)[0]
                proba = hindi_model.predict_proba(vectorized)[0]
                confidence = float(max(proba))
                
                result = "Cyberbullying Detected" if prediction == 1 else "Safe Message"
                
                # Multi-label categories
                categories = {
                    "toxic": int(prediction == 1),
                    "severe_toxic": 0,
                    "obscene": 0,
                    "threat": 0,
                    "insult": 0,
                    "identity_hate": 0
                }
                
                # Analyze for specific categories
                text_lower = input_data.text.lower()
                if prediction == 1:
                    severe_words = ['madarchod', 'behenchod']
                    if any(word in text_lower for word in severe_words):
                        categories["severe_toxic"] = 1
                    
                    insult_words = ['chutiya', 'gandu', 'bewakoof', 'pagal']
                    if any(word in text_lower for word in insult_words):
                        categories["insult"] = 1
                    
                    threat_words = ['maar', 'marunga']
                    if any(word in text_lower for word in threat_words):
                        categories["threat"] = 1
                
                # Extract toxic keywords
                toxic_keywords = []
                if prediction == 1:
                    feature_names = hindi_vectorizer.get_feature_names_out()
                    toxic_keywords = extract_toxic_keywords(
                        input_data.text,
                        vectorized,
                        feature_names
                    )
                
                return {
                    "prediction": result,
                    "confidence": round(confidence, 2),
                    "categories": categories,
                    "toxicKeywords": toxic_keywords,
                    "language": language,
                    "model_used": "hindi_trained_model"
                }
            except Exception as e:
                print(f"Hindi model error: {e}, falling back to keyword matching")
                # Fall back to keyword matching if model fails
                pass
        
        # For Telugu or Hindi fallback, use keyword-based detection
        if language in ['hindi', 'telugu']:
            is_toxic, toxic_words, confidence = check_multilingual_toxicity(
                input_data.text, 
                language
            )
            
            result = "Cyberbullying Detected" if is_toxic else "Safe Message"
            
            # Multi-label categories for non-English
            categories = {
                "toxic": int(is_toxic),
                "severe_toxic": 0,
                "obscene": 0,
                "threat": 0,
                "insult": 0,
                "identity_hate": 0
            }
            
            if is_toxic:
                # Categorize based on keywords
                text_lower = input_data.text.lower()
                
                # Check for severe toxic
                severe_words = ['madarchod', 'behenchod', 'dengey', 'lanjakodaka']
                if any(word in text_lower for word in severe_words):
                    categories["severe_toxic"] = 1
                
                # Check for insults
                insult_words = ['chutiya', 'gandu', 'bewakoof', 'pagal', 'erripuka', 'buddodu']
                if any(word in text_lower for word in insult_words):
                    categories["insult"] = 1
                
                # Check for threats
                threat_words = ['maar', 'marunga', 'kottesta', 'champesta']
                if any(word in text_lower for word in threat_words):
                    categories["threat"] = 1
            
            return {
                "prediction": result,
                "confidence": round(confidence, 2),
                "categories": categories,
                "toxicKeywords": toxic_words,
                "language": language,
                "model_used": "keyword_matching"
            }
        
        # For English, use trained English model
        if not all([model, vectorizer, cleaner]):
            return {
                "error": "English models not loaded",
                "model_status": {
                    "model": model is not None,
                    "vectorizer": vectorizer is not None,
                    "cleaner": cleaner is not None
                }
            }
        
        cleaned_text = cleaner.transform([input_data.text])
        vectorized = vectorizer.transform(cleaned_text)
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        confidence = float(max(proba))
        
        result = "Cyberbullying Detected" if prediction == 1 else "Safe Message"
        
        # Multi-label toxicity categories
        categories = {
            "toxic": int(prediction == 1),
            "severe_toxic": 0,
            "obscene": 0,
            "threat": 0,
            "insult": 0,
            "identity_hate": 0
        }
        
        # Analyze text for specific categories
        text_lower = input_data.text.lower()
        
        if prediction == 1:
            # Severe toxic indicators
            severe_words = ['kill', 'die', 'death']
            if any(word in text_lower for word in severe_words):
                categories["severe_toxic"] = 1
            
            # Obscene indicators
            obscene_words = ['damn', 'hell', 'crap', 'fuck']
            if any(word in text_lower for word in obscene_words):
                categories["obscene"] = 1
            
            # Threat indicators
            threat_words = ['kill you', 'hurt you', 'destroy you', 'gonna get you']
            if any(phrase in text_lower for phrase in threat_words):
                categories["threat"] = 1
            
            # Insult indicators
            insult_words = ['stupid', 'idiot', 'dumb', 'moron', 'fool', 'loser', 'ugly']
            if any(word in text_lower for word in insult_words):
                categories["insult"] = 1
            
            # Identity hate indicators
            hate_words = ['hate', 'racist', 'sexist']
            if any(word in text_lower for word in hate_words):
                categories["identity_hate"] = 1
        
        response = {
            "prediction": result,
            "confidence": round(confidence, 2),
            "categories": categories,
            "language": "english",
            "model_used": "english_trained_model"
        }
        
        # Add toxic keywords if bullying detected
        if prediction == 1:
            feature_names = vectorizer.get_feature_names_out()
            toxic_keywords = extract_toxic_keywords(
                input_data.text, 
                vectorized, 
                feature_names
            )
            response["toxicKeywords"] = toxic_keywords
        
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in predict: {str(e)}")
        print(error_details)
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
            "hindi_vectorizer": hindi_vectorizer is not None
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Cyberbullying Detection ML Service",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }
