# ONNX Syllable Counter

A high-performance, cross-platform syllable counting model exported from TensorFlow to ONNX format.

## 🎯 Model Details

- **Accuracy**: 95.82% on validation set
- **Architecture**: Bidirectional GRU → GRU → Dense
- **Parameters**: 6,833 total
- **Model Size**: ~27 KB
- **Input**: One-hot encoded character sequences (max 18 chars, 28 char vocabulary)
- **Output**: Predicted syllable count (float, round to nearest integer)

## 📁 Files

- `syllable_model.onnx` - The ONNX model file
- `model_metadata.json` - Complete model configuration and metadata
- `python_example.py` - Python usage example with ONNX Runtime
- `javascript_example.js` - Node.js usage example
- `csharp_example.cs` - C# usage example

## 🚀 Quick Start

### Python
```bash
pip install onnxruntime numpy
python python_example.py
```

### JavaScript (Node.js)
```bash
npm install onnxruntime-node
node javascript_example.js
```

### C#
```bash
# Install NuGet packages:
# Microsoft.ML.OnnxRuntime
# Newtonsoft.Json
dotnet run csharp_example.cs
```

## 📊 Character Encoding

Words are encoded as one-hot vectors using this 28-character alphabet:
```
' - a b c d e f g h i j k l m n o p q r s t u v w x y z
```

## 🔧 Usage Pattern

1. **Preprocess**: Convert word to lowercase, truncate to 18 characters
2. **Encode**: Convert to one-hot tensor of shape `[1, 18, 28]`
3. **Predict**: Run through ONNX model
4. **Postprocess**: Round to nearest integer, minimum 1 syllable

## 🌐 Language Support

This ONNX model works with any language that has ONNX Runtime bindings:

- **Python**: onnxruntime
- **JavaScript/Node.js**: onnxruntime-node
- **C#**: Microsoft.ML.OnnxRuntime
- **C++**: onnxruntime C++ API
- **Java**: onnxruntime Java API
- **Rust**: ort crate
- **Go**: onnxruntime-go
- **And many more...**

## 📈 Performance

- **Inference Time**: ~1-5ms per word
- **Memory Usage**: ~10-20MB
- **Throughput**: 1000+ words/second
- **Cold Start**: ~100ms

## 🎯 Accuracy Examples

| Word | Predicted | Actual |
|------|-----------|---------|
| hello | 2 | 2 |
| world | 1 | 1 |
| python | 2 | 2 |
| elixir | 3 | 3 |
| programming | 3 | 3 |
| artificial | 4 | 4 |
| intelligence | 4 | 4 |

## 🔧 Integration

### Web APIs
Use in REST APIs, GraphQL servers, or microservices for real-time syllable counting.

### Desktop Applications
Integrate into native desktop apps with minimal overhead.

### Mobile Apps
Deploy on iOS (Core ML) or Android (TensorFlow Lite conversion).

### Embedded Systems
Run on edge devices with ONNX Runtime for embedded systems.

## 📝 License

Same as original model - check the source repository for licensing terms.

---

**Built from the original TensorFlow syllable counter | Exported with tf2onnx**
