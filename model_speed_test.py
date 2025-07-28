#!/usr/bin/env python3
"""
Simple TensorFlow model inference speed test.
Focuses purely on the neural network performance.
"""

import time

import numpy as np


def main():
    print("🧠 Loading TensorFlow model...")
    start_time = time.time()
    
    from syllable import ModelSyllableCounter
    
    msc = ModelSyllableCounter()
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    
    # Warm up
    print("🔥 Warming up...")
    for _ in range(5):
        msc.count_syllables("test")
    
    print("\n🚀 Model ready! Testing inference speed...")
    print("   Type 'quit' to exit, 'batch' for batch test")
    print("=" * 50)
    
    while True:
        try:
            text = input("\n📝 Word: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            elif text.lower() == 'batch':
                batch_test(msc)
                continue
            elif not text:
                continue
            
            # Multiple runs for more accurate timing
            times = []
            results = []
            
            for _ in range(10):
                start = time.perf_counter()
                result = msc.count_syllables(text)
                end = time.perf_counter()
                times.append((end - start) * 1000)
                results.append(result)
            
            # Check consistency
            unique_results = set(results)
            if len(unique_results) == 1:
                result_str = str(results[0])
            else:
                result_str = f"Inconsistent: {unique_results}"
            
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            print(f"   🎯 Result: {result_str}")
            print(f"   ⚡ Avg: {avg_time:.3f}ms (min: {min_time:.3f}ms, max: {max_time:.3f}ms)")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("👋 Done!")

def batch_test(msc):
    """Test batch processing performance"""
    test_words = [
        "cat", "dog", "elephant", "computer", "artificial", "intelligence",
        "python", "programming", "tensorflow", "machine", "learning", "neural",
        "network", "syllable", "pronunciation", "dictionary", "algorithm"
    ]
    
    print(f"\n🏁 Batch testing {len(test_words)} words...")
    
    # Single words
    total_start = time.perf_counter()
    for word in test_words:
        result = msc.count_syllables(word)
    total_end = time.perf_counter()
    
    total_time = (total_end - total_start) * 1000
    avg_per_word = total_time / len(test_words)
    words_per_second = len(test_words) / (total_time / 1000)
    
    print(f"   📊 Total time: {total_time:.2f}ms")
    print(f"   📏 Average per word: {avg_per_word:.3f}ms")
    print(f"   🚀 Words per second: {words_per_second:.1f}")
    
    # Test longer text
    long_text = " ".join(test_words)
    start = time.perf_counter()
    long_result = msc.count_syllables(long_text)
    end = time.perf_counter()
    
    long_time = (end - start) * 1000
    print(f"   📝 Long text ({len(long_text)} chars): {long_result} in {long_time:.2f}ms")

if __name__ == "__main__":
    main() 