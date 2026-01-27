import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
from nltk.tokenize import sent_tokenize

class NewsSummarizer:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        nltk.download('punkt', quiet=True)
    
    def generate_summary(self, text, max_length=150, min_length=30):
        try:
            input_text = "summarize: " + text
            input_ids = self.tokenizer.encode(
                input_text, return_tensors='pt', max_length=512, truncation=True
            )
            with torch.no_grad():
                summary_ids = self.model.generate(
                    input_ids, max_length=max_length, min_length=min_length,
                    length_penalty=2.0, num_beams=4, early_stopping=True
                )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return {
                'text': summary,
                'original_length': len(text.split()),
                'summary_length': len(summary.split()),
                'compression_ratio': round((1 - len(summary.split()) / len(text.split())) * 100, 2)
            }
        except Exception as e:
            return {'text': 'Error generating summary', 'error': str(e)}
    
    def extractive_summary(self, text, num_sentences=3):
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        indices = [0, len(sentences)//2, -1]
        selected_sentences = [sentences[i] for i in indices[:num_sentences]]
        return ' '.join(selected_sentences)
