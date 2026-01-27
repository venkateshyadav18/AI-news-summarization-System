import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import re

class SentimentAnalyzer:
    def __init__(self, model_name='cardiffnlp/twitter-roberta-base-sentiment-latest'):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.use_transformer = True
        except:
            self.use_transformer = False
            print("Using TextBlob as fallback for sentiment analysis")
    
    def analyze_sentiment(self, text):
        try:
            if self.use_transformer:
                return self._transformer_sentiment(text)
            else:
                return self._textblob_sentiment(text)
        except Exception:
            return self._textblob_sentiment(text)
    
    def _transformer_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        scores = predictions[0].tolist()
        labels = ['negative', 'neutral', 'positive']
        sentiment_scores = dict(zip(labels, scores))
        predicted_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        confidence = max(scores) * 100
        return {
            'sentiment': predicted_sentiment,
            'confidence': round(confidence, 2),
            'scores': {
                'positive': round(sentiment_scores.get('positive', 0) * 100, 2),
                'negative': round(sentiment_scores.get('negative', 0) * 100, 2),
                'neutral': round(sentiment_scores.get('neutral', 0) * 100, 2)
            }
        }
    
    def _textblob_sentiment(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        positive_score = max(0, polarity * 50 + 50) if polarity > 0 else 20
        negative_score = max(0, -polarity * 50 + 50) if polarity < 0 else 20
        neutral_score = 100 - positive_score - negative_score
        confidence = abs(polarity) * 100 if abs(polarity) > 0.1 else 60
        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 2),
            'scores': {
                'positive': round(positive_score, 2),
                'negative': round(negative_score, 2),
                'neutral': round(neutral_score, 2)
            }
        }
