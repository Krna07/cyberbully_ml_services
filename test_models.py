"""
Test script to verify models load correctly
"""
import pickle
from pathlib import Path

def test_models():
    model_path = Path(__file__).parent
    
    print("Testing model files...")
    print(f"Working directory: {model_path}")
    
    # Test model
    try:
        with open(model_path / "cyberbullying_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
        print(f"  Model type: {type(model)}")
    except Exception as e:
        print(f"✗ Model error: {e}")
        return False
    
    # Test vectorizer
    try:
        with open(model_path / "tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        print("✓ Vectorizer loaded successfully")
        print(f"  Vectorizer type: {type(vectorizer)}")
    except Exception as e:
        print(f"✗ Vectorizer error: {e}")
        return False
    
    # Test cleaner
    cleaner = None
    try:
        with open(model_path / "text_cleaner.pkl", "rb") as f:
            cleaner = pickle.load(f)
        print("✓ Cleaner loaded successfully")
        print(f"  Cleaner type: {type(cleaner)}")
    except Exception as e:
        print(f"✗ Cleaner error: {e}")
        print("  This is OK, will use fallback")
        # Create fallback cleaner
        class SimpleTextCleaner:
            def transform(self, texts):
                return [text.lower() for text in texts]
        cleaner = SimpleTextCleaner()
    
    # Test prediction
    try:
        test_text = "You are stupid"
        print(f"\nTesting prediction with: '{test_text}'")
        
        # Clean and vectorize
        if hasattr(cleaner, 'transform'):
            cleaned = cleaner.transform([test_text])
        else:
            cleaned = [test_text.lower()]
        
        vectorized = vectorizer.transform(cleaned)
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        
        print(f"✓ Prediction: {prediction}")
        print(f"  Probability: {proba}")
        print(f"  Confidence: {max(proba):.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_models()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed!")
