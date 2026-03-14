import pickle
import re

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return ' '.join(text.split())

with open('cyberbullyingcommmon_model.pkl', 'rb') as f:
    model = pickle.load(f)
print('Model type:', type(model))
print('Classes:', model.classes_ if hasattr(model, 'classes_') else 'N/A')
print('Params:', model.get_params() if hasattr(model, 'get_params') else 'N/A')

with open('tfidf_vectorizercommon.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
print('Vectorizer type:', type(vectorizer))
print('Vocab size:', len(vectorizer.get_feature_names_out()))

tests = ['you are so stupid and ugly', 'hello how are you', 'I will kill you', 'you idiot loser']
for t in tests:
    cleaned = clean(t)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0] if hasattr(model, 'predict_proba') else None
    print(f'Text: "{t}" => pred={pred}, proba={proba}')
