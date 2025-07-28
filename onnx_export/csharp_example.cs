// Example: Using the ONNX syllable counting model in C#
// Install-Package Microsoft.ML.OnnxRuntime

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public class SyllableCounter : IDisposable
{
    private InferenceSession session;
    private string[] chars;
    private int maxLen;
    private Dictionary<char, int> charToIdx;
    
    public SyllableCounter(string modelPath = "syllable_model.onnx", string metadataPath = "model_metadata.json")
    {
        // Load ONNX model
        session = new InferenceSession(modelPath);
        
        // Load metadata
        var metadataJson = File.ReadAllText(metadataPath);
        dynamic metadata = JsonConvert.DeserializeObject(metadataJson);
        
        chars = ((JArray)metadata.character_encoding.alphabet).ToObject<string[]>();
        maxLen = metadata.character_encoding.max_word_length;
        charToIdx = chars.Select((c, i) => new { c, i }).ToDictionary(x => x.c[0], x => x.i);
        
        Console.WriteLine($"Loaded syllable counter with {chars.Length} characters, max length {maxLen}");
    }
    
    private DenseTensor<float> EncodeWord(string word)
    {
        word = word.ToLower();
        if (word.Length > maxLen) word = word.Substring(0, maxLen);
        
        var tensor = new DenseTensor<float>(new[] { 1, maxLen, chars.Length });
        
        for (int i = 0; i < word.Length; i++)
        {
            if (charToIdx.TryGetValue(word[i], out int charIdx))
            {
                tensor[0, i, charIdx] = 1.0f;
            }
        }
        
        return tensor;
    }
    
    public int CountSyllables(string word)
    {
        if (string.IsNullOrWhiteSpace(word)) return 0;
        
        var encoded = EncodeWord(word);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(session.InputMetadata.Keys.First(), encoded)
        };
        
        using var results = session.Run(inputs);
        var output = results.First().AsEnumerable<float>().First();
        
        return Math.Max(1, (int)Math.Round(output));
    }
    
    public List<int> CountText(string text)
    {
        var words = text.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        return words.Select(CountSyllables).ToList();
    }
    
    public void Dispose()
    {
        session?.Dispose();
    }
}

// Example usage
class Program
{
    static void Main()
    {
        using var counter = new SyllableCounter();
        
        // Test individual words
        var testWords = new[]
        {
            "hello", "world", "csharp", "elixir", "programming",
            "artificial", "intelligence", "supercalifragilisticexpialidocious"
        };
        
        Console.WriteLine("Syllable counting with ONNX model:");
        foreach (var word in testWords)
        {
            var syllables = counter.CountSyllables(word);
            Console.WriteLine($"  {word}: {syllables} syllables");
        }
        
        // Test sentences
        var sentences = new[]
        {
            "Hello world",
            "C# is a programming language",
            "Machine learning with neural networks"
        };
        
        Console.WriteLine("\nSentence syllable counts:");
        foreach (var sentence in sentences)
        {
            var counts = counter.CountText(sentence);
            var total = counts.Sum();
            Console.WriteLine($"  '{sentence}': [{string.Join(", ", counts)}] (total: {total})");
        }
    }
}
