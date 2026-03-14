"""
Test script to verify Hindi model works correctly
"""
import pickle
from pathlib import Path

def test_hindi_model():
    model_path = Path(__file__).parent
    
    print("Testing Hindi model files...")
    print(f"Working directory: {model_path}\n")
    
    # Test Hindi model
    try:
        with open(model_path / "hindi_cyberbullying_model_v2.pkl", "rb") as f:
            hindi_model = pickle.load(f)
        print("✓ Hindi model loaded successfully")
        print(f"  Model type: {type(hindi_model)}")
    except Exception as e:
        print(f"✗ Hindi model error: {e}")
        return False
    
    # Test Hindi vectorizer
    try:
        with open(model_path / "hindi_vectorizer_v2.pkl", "rb") as f:
            hindi_vectorizer = pickle.load(f)
        print("✓ Hindi vectorizer loaded successfully")
        print(f"  Vectorizer type: {type(hindi_vectorizer)}")
    except Exception as e:
        print(f"✗ Hindi vectorizer error: {e}")
        return False
    
    # Test prediction with Hindi text
    try:
        test_texts = [
            "tu behenchod hai",
            "namaste kaise ho",
            "madarchod sala"
        ]
        
        print(f"\nTesting predictions:")
        for text in test_texts:
            print(f"\n  Text: '{text}'")
            
            # Simple cleaning (lowercase)
            cleaned = [text.lower()]
            
            # Vectorize
            vectorized = hindi_vectorizer.transform(cleaned)
            
            # Predict
            prediction = hindi_model.predict(vectorized)[0]
            proba = hindi_model.predict_proba(vectorized)[0]
            confidence = max(proba)
            
            result = "Cyberbullying" if prediction == 1 else "Safe"
            print(f"  Prediction: {result}")
            print(f"  Confidence: {confidence:.2f}")
        
        return True
    except Exception as e:
        print(f"\n✗ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hindi_model()
    if success:
        print("\n✓ All Hindi model tests passed!")
    else:
        print("\n✗ Hindi model tests failed!")
