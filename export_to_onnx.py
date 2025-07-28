#!/usr/bin/env python3
"""
Export the TensorFlow syllable counting model to ONNX format.

This creates a cross-platform ONNX model that can be used in Elixir via ONNX Runtime,
C++, JavaScript, or any other language that supports ONNX.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def export_to_onnx():
    """Export the TensorFlow model to ONNX format."""
    
    model_dir = Path('./syllable/model_data/model')
    chars_file = Path('./syllable/model_data/chars.json')
    output_dir = Path('./onnx_export')
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("🔄 Loading TensorFlow model...")
    model = tf.keras.models.load_model(model_dir)
    
    # Load character config
    with open(chars_file) as f:
        char_config = json.load(f)
    
    print("\nModel summary:")
    model.summary()
    
    # Test the model with sample input
    print("\n🧪 Testing model with sample input...")
    test_input = np.random.random((1, 18, 28)).astype(np.float32)
    test_output = model.predict(test_input, verbose=0)
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")
    print(f"Test output value: {test_output[0][0]}")
    
    try:
        import onnx
        import tf2onnx
        
        print("\n🔄 Converting to ONNX...")
        
        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            opset=13,  # Use a compatible ONNX opset
            output_path=str(output_dir / "syllable_model.onnx")
        )
        
        print("✅ Successfully exported to ONNX!")
        
        # Verify the ONNX model
        print("\n🔍 Verifying ONNX model...")
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model is valid!")
        
        # Test ONNX model
        print("\n🧪 Testing ONNX model...")
        test_onnx_model(output_dir / "syllable_model.onnx", test_input, test_output)
        
        # Save metadata
        metadata = {
            "model_type": "syllable_counter",
            "version": "1.0.0",
            "char_config": char_config,
            "input_shape": [1, 18, 28],
            "output_shape": [1, 1],
            "description": "Neural network for counting syllables in English words",
            "usage": {
                "input": "One-hot encoded character sequences (max 18 chars, 28 character vocab)",
                "output": "Predicted syllable count (float, should be rounded to nearest integer)",
                "characters": char_config["chars"],
                "max_length": char_config["maxlen"]
            }
        }
        
        with open(output_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📁 Files saved to {output_dir}:")
        print(f"   - syllable_model.onnx ({(output_dir / 'syllable_model.onnx').stat().st_size / 1024:.1f} KB)")
        print(f"   - model_metadata.json ({(output_dir / 'model_metadata.json').stat().st_size / 1024:.1f} KB)")
        
        # Create usage example
        create_usage_examples(output_dir)
        
    except ImportError as e:
        print(f"\n❌ Missing dependencies for ONNX export: {e}")
        print("\nTo install required packages:")
        print("   pip install tf2onnx onnx onnxruntime")
        return False
    
    return True

def test_onnx_model(onnx_path, test_input, expected_output):
    """Test the exported ONNX model."""
    
    try:
        import onnxruntime as ort

        # Create ONNX Runtime session
        session = ort.InferenceSession(str(onnx_path))
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"   Input name: {input_name}")
        print(f"   Output name: {output_name}")
        
        # Run inference
        onnx_output = session.run([output_name], {input_name: test_input})
        
        # Compare outputs
        tf_result = expected_output[0][0]
        onnx_result = onnx_output[0][0][0]
        difference = abs(tf_result - onnx_result)
        
        print(f"   TensorFlow output: {tf_result}")
        print(f"   ONNX output: {onnx_result}")
        print(f"   Difference: {difference}")
        
        if difference < 1e-5:
            print("   ✅ ONNX model matches TensorFlow model!")
        else:
            print("   ⚠️  ONNX model differs from TensorFlow model")
            
    except ImportError:
        print("   ⚠️  onnxruntime not available, skipping ONNX test")

def create_usage_examples(output_dir):
    """Create usage examples for different languages."""
    
    # Python example
    python_example = '''#!/usr/bin/env python3
"""
Example usage of the exported ONNX syllable counting model.
"""

import numpy as np
import onnxruntime as ort
import json

# Load the model
session = ort.InferenceSession("syllable_model.onnx")

# Load metadata
with open("model_metadata.json") as f:
    metadata = json.load(f)

chars = metadata["char_config"]["chars"]
max_len = metadata["char_config"]["maxlen"]
char_to_idx = {c: i for i, c in enumerate(chars)}

def encode_word(word):
    """Encode a word to one-hot tensor."""
    word = word.lower()[:max_len]  # Truncate and lowercase
    
    # Create zero tensor
    encoded = np.zeros((1, max_len, len(chars)), dtype=np.float32)
    
    # Set one-hot values
    for i, char in enumerate(word):
        if char in char_to_idx:
            encoded[0, i, char_to_idx[char]] = 1.0
    
    return encoded

def count_syllables(word):
    """Count syllables in a word using the ONNX model."""
    encoded = encode_word(word)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: encoded})
    
    # Round to nearest integer, minimum 1
    syllables = max(1, round(result[0][0][0]))
    return syllables

# Test the model
test_words = ["hello", "world", "python", "elixir", "supercalifragilisticexpialidocious"]

for word in test_words:
    syllables = count_syllables(word)
    print(f"{word}: {syllables} syllables")
'''
    
    # JavaScript example
    js_example = '''// Example usage in Node.js with onnxruntime-node
// npm install onnxruntime-node

const ort = require('onnxruntime-node');
const fs = require('fs');

async function loadModel() {
    const session = await ort.InferenceSession.create('./syllable_model.onnx');
    const metadata = JSON.parse(fs.readFileSync('./model_metadata.json', 'utf8'));
    
    const chars = metadata.char_config.chars;
    const maxLen = metadata.char_config.maxlen;
    const charToIdx = Object.fromEntries(chars.map((c, i) => [c, i]));
    
    function encodeWord(word) {
        word = word.toLowerCase().slice(0, maxLen);
        
        const encoded = new Float32Array(1 * maxLen * chars.length);
        
        for (let i = 0; i < word.length; i++) {
            const char = word[i];
            if (char in charToIdx) {
                const charIdx = charToIdx[char];
                const flatIdx = i * chars.length + charIdx;
                encoded[flatIdx] = 1.0;
            }
        }
        
        return new ort.Tensor('float32', encoded, [1, maxLen, chars.length]);
    }
    
    async function countSyllables(word) {
        const encoded = encodeWord(word);
        const feeds = { [session.inputNames[0]]: encoded };
        const results = await session.run(feeds);
        const output = results[session.outputNames[0]];
        
        return Math.max(1, Math.round(output.data[0]));
    }
    
    // Test the model
    const testWords = ['hello', 'world', 'javascript', 'elixir'];
    
    for (const word of testWords) {
        const syllables = await countSyllables(word);
        console.log(`${word}: ${syllables} syllables`);
    }
}

loadModel().catch(console.error);
'''
    
    # Save examples
    with open(output_dir / "example_python.py", 'w') as f:
        f.write(python_example)
    
    with open(output_dir / "example_javascript.js", 'w') as f:
        f.write(js_example)
    
    # Create README
    readme = '''# ONNX Syllable Counter

This directory contains the exported ONNX model for syllable counting.

## Files

- `syllable_model.onnx` - The ONNX model file
- `model_metadata.json` - Model configuration and character mapping
- `example_python.py` - Python usage example
- `example_javascript.js` - JavaScript usage example

## Model Details

- **Input**: Float32 tensor of shape [1, 18, 28]
  - 1: batch size (single word)
  - 18: maximum word length (characters)
  - 28: character vocabulary size (one-hot encoded)

- **Output**: Float32 tensor of shape [1, 1]
  - Single float value representing predicted syllable count
  - Should be rounded to nearest integer, minimum 1

## Character Encoding

Words are encoded as one-hot vectors using this alphabet:
`'`, `-`, `a`, `b`, `c`, ..., `z` (28 characters total)

## Usage in Different Languages

### Python
```bash
pip install onnxruntime numpy
python example_python.py
```

### JavaScript (Node.js)
```bash
npm install onnxruntime-node
node example_javascript.js
```

### Elixir (with ONNX Runtime bindings)
```elixir
# Note: Requires ONNX Runtime Elixir bindings (external library)
# Consider using the native Axon implementation instead
```

### C++, C#, Java, etc.
The ONNX model can be used with ONNX Runtime in any supported language.
See: https://onnxruntime.ai/docs/get-started/

## Performance

- Model size: ~27 KB
- Inference time: ~1-5ms per word
- Accuracy: 95.8% on validation set
'''
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme)
    
    print("   - example_python.py")
    print("   - example_javascript.js") 
    print("   - README.md")

if __name__ == "__main__":
    success = export_to_onnx()
    if success:
        print("\n🎉 ONNX export completed successfully!")
        print("\nYou now have two options:")
        print("1. 🦾 Use the native Elixir/Axon implementation (recommended)")
        print("2. 🌐 Use the ONNX model with ONNX Runtime bindings")
    else:
        print("\n💡 Consider using the native Elixir/Axon implementation instead") 