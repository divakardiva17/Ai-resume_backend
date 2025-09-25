import spacy

# Load Spacy model for keyword extraction
nlp = spacy.load('en_core_web_sm')

# Extract keywords (e.g., Nouns, Named Entities)
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ == 'NOUN']
    return keywords
