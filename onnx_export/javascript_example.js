// Example: Using the ONNX syllable counting model in Node.js
// npm install onnxruntime-node

const ort = require('onnxruntime-node');
const fs = require('fs');

class SyllableCounter {
    constructor(modelPath = './syllable_model.onnx', metadataPath = './model_metadata.json') {
        this.modelPath = modelPath;
        this.metadataPath = metadataPath;
        this.session = null;
        this.metadata = null;
        this.chars = null;
        this.maxLen = null;
        this.charToIdx = null;
    }
    
    async initialize() {
        // Load ONNX model
        this.session = await ort.InferenceSession.create(this.modelPath);
        
        // Load metadata
        this.metadata = JSON.parse(fs.readFileSync(this.metadataPath, 'utf8'));
        this.chars = this.metadata.character_encoding.alphabet;
        this.maxLen = this.metadata.character_encoding.max_word_length;
        this.charToIdx = Object.fromEntries(this.chars.map((c, i) => [c, i]));
        
        console.log(`Loaded syllable counter with ${this.chars.length} characters, max length ${this.maxLen}`);
    }
    
    encodeWord(word) {
        word = word.toLowerCase().slice(0, this.maxLen);
        
        const encoded = new Float32Array(1 * this.maxLen * this.chars.length);
        
        for (let i = 0; i < word.length; i++) {
            const char = word[i];
            if (char in this.charToIdx) {
                const charIdx = this.charToIdx[char];
                const flatIdx = i * this.chars.length + charIdx;
                encoded[flatIdx] = 1.0;
            }
        }
        
        return new ort.Tensor('float32', encoded, [1, this.maxLen, this.chars.length]);
    }
    
    async countSyllables(word) {
        if (!word || !word.trim()) return 0;
        
        const encoded = this.encodeWord(word);
        const feeds = { [this.session.inputNames[0]]: encoded };
        const results = await this.session.run(feeds);
        const output = results[this.session.outputNames[0]];
        
        return Math.max(1, Math.round(output.data[0]));
    }
    
    async countText(text) {
        const words = text.split(/\s+/).filter(w => w.length > 0);
        const results = [];
        
        for (const word of words) {
            results.push(await this.countSyllables(word));
        }
        
        return results;
    }
}

// Example usage
async function main() {
    const counter = new SyllableCounter();
    await counter.initialize();
    
    // Test individual words
    const testWords = [
        'hello', 'world', 'javascript', 'elixir', 'programming',
        'artificial', 'intelligence', 'supercalifragilisticexpialidocious'
    ];
    
    console.log('\nSyllable counting with ONNX model:');
    for (const word of testWords) {
        const syllables = await counter.countSyllables(word);
        console.log(`  ${word}: ${syllables} syllables`);
    }
    
    // Test sentences
    const sentences = [
        'Hello world',
        'JavaScript is a programming language',
        'Machine learning with neural networks'
    ];
    
    console.log('\nSentence syllable counts:');
    for (const sentence of sentences) {
        const counts = await counter.countText(sentence);
        const total = counts.reduce((a, b) => a + b, 0);
        console.log(`  '${sentence}': ${JSON.stringify(counts)} (total: ${total})`);
    }
}

main().catch(console.error);
