import json
import numpy as np
import re

class NLPPreprocessor:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.intent_to_idx = {}
        self.idx_to_intent = {}
        self.vocab_size = 0
        
    def clean_text(self, text):
        """Lowercase and remove punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
        
    def build_vocab(self, dataset_path):
        """Reads the dataset, builds word vocabulary and intent mappings."""
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            
        intents = data['intents']
        
        # Build Intent Mapping
        for idx, intent in enumerate(intents.keys()):
            self.intent_to_idx[intent] = idx
            self.idx_to_intent[idx] = intent
            
        # Build Word Vocabulary
        # Add an <UNK> token for unknown words
        self.vocab['<UNK>'] = 0
        self.inverse_vocab[0] = '<UNK>'
        word_idx = 1
        
        for intent, phrases in intents.items():
            for phrase in phrases:
                words = self.clean_text(phrase).split()
                for word in words:
                    if word not in self.vocab:
                        self.vocab[word] = word_idx
                        self.inverse_vocab[word_idx] = word
                        word_idx += 1
                        
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary built with {self.vocab_size} words.")
        print(f"Intents: {list(self.intent_to_idx.keys())}")
        
    def text_to_sequence(self, text):
        """Converts text into a list of word indices."""
        words = self.clean_text(text).split()
        seq = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        # Avoid empty sequences
        if len(seq) == 0:
            seq = [self.vocab['<UNK>']]
        return seq

    def sequence_to_onehot(self, seq):
        """
        Converts a sequence of indices into a list of one-hot vectors.
        Each vector has shape (vocab_size, 1) to be compatible with pure NumPy LSTM.
        """
        onehots = []
        for idx in seq:
            vec = np.zeros((self.vocab_size, 1))
            vec[idx, 0] = 1.0
            onehots.append(vec)
        return onehots

    def process_dataset(self, dataset_path):
        """Returns X (list of one-hot sequences) and Y (list of target intents)."""
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            
        intents = data['intents']
        X = []
        Y = []
        
        for intent, phrases in intents.items():
            target_idx = self.intent_to_idx[intent]
            for phrase in phrases:
                seq = self.text_to_sequence(phrase)
                onehot_seq = self.sequence_to_onehot(seq)
                X.append(onehot_seq)
                Y.append(target_idx)
                
        return X, Y

if __name__ == '__main__':
    # Test the preprocessor
    prep = NLPPreprocessor()
    prep.build_vocab('dataset.json')
    X, Y = prep.process_dataset('dataset.json')
    print(f"Processed {len(X)} samples.")
    print(f"Sample 0 Sequence length: {len(X[0])}, Target Intent: {Y[0]}")
    print(f"Input vector shape: {X[0][0].shape}")
