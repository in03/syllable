#!/usr/bin/env python3
"""
Interactive syllable counting performance test.
Loads the model once, then allows fast interactive testing.
"""

import statistics
import time
from typing import List


def main():
    print("🔄 Loading syllable counters (this is the slow part)...")
    start_load = time.time()
    
    # Import and initialize - this is the slow part
    from syllable import (
        CmudictSyllableCounter,
        CompositeSyllableCounter,
        ModelSyllableCounter,
    )
    
    print("   📚 Loading CMUdict...")
    csc = CmudictSyllableCounter()
    
    print("   🧠 Loading TensorFlow model...")
    msc = ModelSyllableCounter()
    
    print("   🔗 Setting up composite counter...")
    comp = CompositeSyllableCounter([csc, msc])
    
    load_time = time.time() - start_load
    print(f"✅ Loaded in {load_time:.2f} seconds")
    print()
    
    # Warm up the model with a test inference
    print("🔥 Warming up model...")
    warm_start = time.time()
    _ = comp.count_syllables("test")
    warm_time = time.time() - warm_start
    print(f"   First inference: {warm_time*1000:.1f}ms")
    print()
    
    print("🚀 Ready for interactive testing!")
    print("   Type words/sentences to test syllable counting speed")
    print("   Commands: 'quit' to exit, 'bench' for benchmark")
    print("=" * 60)
    
    inference_times = []
    
    while True:
        try:
            text = input("\n📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            elif text.lower() == 'bench':
                run_benchmark(comp)
                continue
            elif text.lower() == 'stats' and inference_times:
                show_stats(inference_times)
                continue
            elif not text:
                continue
                
            # Time the inference
            start_time = time.time()
            result = comp.count_syllables(text)
            end_time = time.time()
            
            inference_ms = (end_time - start_time) * 1000
            inference_times.append(inference_ms)
            
            # Show results
            print(f"   🎯 Result: {result}")
            print(f"   ⚡ Speed: {inference_ms:.2f}ms")
            
            # Also show individual counter results for comparison
            cmu_result = csc.count_syllables(text)
            model_result = msc.count_syllables(text)
            print(f"   📚 CMUdict: {cmu_result}")
            print(f"   🧠 Model: {model_result}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n📊 Session summary:")
    if inference_times:
        show_stats(inference_times)
    print("👋 Goodbye!")

def run_benchmark(comp):
    """Run a benchmark with common words"""
    test_words = [
        "hello", "world", "python", "tensorflow", "artificial", "intelligence",
        "syllable", "counting", "performance", "benchmark", "optimization",
        "family", "computer", "programming", "development", "application"
    ]
    
    print("\n🏁 Running benchmark with common words...")
    times = []
    
    for word in test_words:
        start = time.time()
        result = comp.count_syllables(word)
        end = time.time()
        
        ms = (end - start) * 1000
        times.append(ms)
        print(f"   {word:12} → {result} ({ms:.2f}ms)")
    
    print("\n📈 Benchmark results:")
    show_stats(times)

def show_stats(times: List[float]):
    """Show statistics for inference times"""
    if not times:
        print("   No data yet")
        return
        
    print(f"   📊 Inferences: {len(times)}")
    print(f"   ⚡ Average: {statistics.mean(times):.2f}ms")
    print(f"   📏 Median: {statistics.median(times):.2f}ms")
    print(f"   🏃 Fastest: {min(times):.2f}ms")
    print(f"   🐌 Slowest: {max(times):.2f}ms")
    if len(times) > 1:
        print(f"   📐 Std dev: {statistics.stdev(times):.2f}ms")

if __name__ == "__main__":
    main() 