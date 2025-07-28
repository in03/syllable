#!/usr/bin/env python3
"""
Example: Using the ONNX syllable counting model in Python
"""

import numpy as np
import onnxruntime as ort
import json

class SyllableCounter:
    def __init__(self, model_path="syllable_model.onnx", metadata_path="model_metadata.json"):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        self.chars = metadata["character_encoding"]["alphabet"]
        self.max_len = metadata["character_encoding"]["max_word_length"]
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
    
    def encode_word(self, word):
        """Encode a word to one-hot tensor."""
        word = word.lower()[:self.max_len]  # Lowercase and truncate
        
        # Create zero tensor
        encoded = np.zeros((1, self.max_len, len(self.chars)), dtype=np.float32)
        
        # Set one-hot values
        for i, char in enumerate(word):
            if char in self.char_to_idx:
                encoded[0, i, self.char_to_idx[char]] = 1.0
        
        return encoded
    
    def count_syllables(self, word):
        """Count syllables in a word."""
        if not word or not word.strip():
            return 0
            
        encoded = self.encode_word(word)
        
        # Run inference
        result = self.session.run([self.output_name], {self.input_name: encoded})
        
        # Round to nearest integer, minimum 1
        syllables = max(1, round(float(result[0][0][0])))
        return syllables
    
    def count_text(self, text):
        """Count syllables in text (multiple words)."""
        words = text.split()
        return [self.count_syllables(word) for word in words]

# Example usage
if __name__ == "__main__":
    counter = SyllableCounter()
    
    # Test individual words
    test_words = [
        "hello", "world", "python", "elixir", "programming",
        "artificial", "intelligence", "supercalifragilisticexpialidocious"
    ]
    
    print("Syllable counting with ONNX model:")
    for word in test_words:
        syllables = counter.count_syllables(word)
        print(f"  {word}: {syllables} syllables")
    
    # Test sentences
    sentences = [
        "Hello world",
        "Python is a programming language", 
        "Machine learning with neural networks"
    ]
    
    print("\nSentence syllable counts:")
    for sentence in sentences:
        counts = counter.count_text(sentence)
        total = sum(counts)
        print(f"  '{sentence}': {counts} (total: {total})")
