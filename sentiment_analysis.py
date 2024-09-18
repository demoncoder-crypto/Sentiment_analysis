from transformers import pipeline

# Load the sentiment-analysis pipeline from Hugging Face
classifier = pipeline('sentiment-analysis')

def analyze_sentiment(text):
    """Classify the sentiment of the text using BERT."""
    result = classifier(text)[0]
    return result['label'], result['score']