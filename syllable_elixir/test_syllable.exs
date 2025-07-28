#!/usr/bin/env elixir

# Interactive test script for the Elixir syllable counter
# Usage: cd syllable_elixir && elixir test_syllable.exs

Mix.install([
  {:syllable_elixir, path: "."},
])

defmodule SyllableTest do
  def run do
    IO.puts("🧪 Testing Elixir Syllable Counter")
    IO.puts(String.duplicate("=", 50))

    # Start the application
    IO.puts("🔄 Starting application...")
    case Application.ensure_all_started(:syllable_elixir) do
      {:ok, _apps} ->
        IO.puts("✅ Application started successfully")
      {:error, reason} ->
        IO.puts("❌ Failed to start application: #{inspect(reason)}")
        exit(1)
    end

    # Wait for services to initialize
    IO.puts("⏳ Waiting for services to initialize...")
    :timer.sleep(2000)

    # Test dictionary lookups
    test_dictionary_words()

    # Test model predictions
    test_model_words()

    # Test sentences
    test_sentences()

    # Show performance stats
    show_stats()

    # Interactive testing
    interactive_test()
  end

  defp test_dictionary_words do
    IO.puts("\n📚 Testing dictionary words...")

    test_words = [
      {"hello", 2},
      {"world", 1},
      {"python", 2},
      {"elixir", 3},
      {"programming", 3},
      {"beautiful", 3},
      {"artificial", 4},
      {"intelligence", 4}
    ]

    Enum.each(test_words, fn {word, expected} ->
      case SyllableElixir.count_detailed(word) do
        {:ok, count, method} ->
          status = if count == expected, do: "✅", else: "❌"
          IO.puts("   #{status} #{word}: #{count} syllables (#{method}) [expected: #{expected}]")

        {:error, reason} ->
          IO.puts("   ❌ #{word}: ERROR - #{inspect(reason)}")
      end
    end)
  end

  defp test_model_words do
    IO.puts("\n🧠 Testing model prediction words (uncommon/made-up)...")

    # These words likely won't be in the dictionary
    test_words = [
      "xyzzyx",     # Made up word
      "blogosphere", # Newer word
      "cryptocurrency", # Technical term
      "supercalifragilisticexpialidocious" # Very long word
    ]

    Enum.each(test_words, fn word ->
      case SyllableElixir.count_detailed(word) do
        {:ok, count, method} ->
          IO.puts("   📊 #{word}: #{count} syllables (#{method})")

        {:error, reason} ->
          IO.puts("   ❌ #{word}: ERROR - #{inspect(reason)}")
      end
    end)
  end

  defp test_sentences do
    IO.puts("\n📝 Testing sentences...")

    test_sentences = [
      "Hello world",
      "The quick brown fox jumps over the lazy dog",
      "Elixir is a beautiful programming language",
      "Machine learning with Axon is amazing"
    ]

    Enum.each(test_sentences, fn sentence ->
      case SyllableElixir.count_detailed(sentence) do
        {:ok, word_details} when is_list(word_details) ->
          total = word_details |> Enum.map(&Map.get(&1, :syllables, 0)) |> Enum.sum()
          IO.puts("   📄 \"#{sentence}\"")
          IO.puts("      Total syllables: #{total}")

          word_details
          |> Enum.each(fn details ->
            word = details.word
            syllables = details.syllables
            method = details.method
            IO.puts("         #{word}: #{syllables} (#{method})")
          end)

        {:error, reason} ->
          IO.puts("   ❌ \"#{sentence}\": ERROR - #{inspect(reason)}")
      end
    end)
  end

  defp show_stats do
    IO.puts("\n📊 Performance Statistics:")

    case SyllableElixir.stats() do
      {:ok, stats} ->
        IO.puts("   Total requests: #{stats.total_requests}")
        IO.puts("   Dictionary hits: #{stats.dictionary_hits}")
        IO.puts("   Model predictions: #{stats.model_predictions}")
        IO.puts("   Errors: #{stats.errors}")
        IO.puts("   Average response time: #{:io_lib.format('~.2f', [stats.avg_response_time_ms])}ms")

        if stats.total_requests > 0 do
          dict_rate = stats.dictionary_hits / stats.total_requests * 100
          model_rate = stats.model_predictions / stats.total_requests * 100
          IO.puts("   Dictionary hit rate: #{:io_lib.format('~.1f', [dict_rate])}%")
          IO.puts("   Model usage rate: #{:io_lib.format('~.1f', [model_rate])}%")
        end

      {:error, reason} ->
        IO.puts("   ❌ Failed to get stats: #{inspect(reason)}")
    end
  end

  defp interactive_test do
    IO.puts("\n🎮 Interactive Testing")
    IO.puts("Enter words or sentences to test (or 'quit' to exit):")

    interactive_loop()
  end

  defp interactive_loop do
    input = IO.gets("📝 > ") |> String.trim()

    case input do
      "" -> interactive_loop()
      "quit" ->
        IO.puts("👋 Goodbye!")
        :ok
      "exit" ->
        IO.puts("👋 Goodbye!")
        :ok
      "stats" ->
        show_stats()
        interactive_loop()
      "benchmark" ->
        run_benchmark()
        interactive_loop()
      text ->
        test_input(text)
        interactive_loop()
    end
  end

  defp test_input(text) do
    start_time = System.monotonic_time(:millisecond)

    case SyllableElixir.count_detailed(text) do
      {:ok, count, method} when is_integer(count) ->
        end_time = System.monotonic_time(:millisecond)
        time_ms = end_time - start_time
        IO.puts("   🎯 Result: #{count} syllables (#{method}) - #{time_ms}ms")

      {:ok, word_details} when is_list(word_details) ->
        end_time = System.monotonic_time(:millisecond)
        time_ms = end_time - start_time

        total = word_details |> Enum.map(&Map.get(&1, :syllables, 0)) |> Enum.sum()
        IO.puts("   🎯 Total: #{total} syllables - #{time_ms}ms")

        word_details
        |> Enum.each(fn details ->
          word = details.word
          syllables = Map.get(details, :syllables, "ERROR")
          method = Map.get(details, :method, Map.get(details, :error, "unknown"))
          IO.puts("      #{word}: #{syllables} (#{method})")
        end)

      {:error, reason} ->
        IO.puts("   ❌ Error: #{inspect(reason)}")
    end
  end

  defp run_benchmark do
    IO.puts("\n🏁 Running benchmark...")

    case SyllableElixir.benchmark() do
      {:ok, results} ->
        IO.puts("   📈 Benchmark Results:")
        IO.puts("      Words tested: #{results.words_tested}")
        IO.puts("      Total time: #{results.total_time_ms}ms")
        IO.puts("      Average time per word: #{:io_lib.format('~.2f', [results.avg_time_ms])}ms")
        IO.puts("      Dictionary hits: #{results.dictionary_hits}")
        IO.puts("      Model predictions: #{results.model_predictions}")
        IO.puts("      Errors: #{results.errors}")
        IO.puts("      Dictionary hit rate: #{:io_lib.format('~.1f', [results.dictionary_hit_rate * 100])}%")
        IO.puts("      Model usage rate: #{:io_lib.format('~.1f', [results.model_usage_rate * 100])}%")

      {:error, reason} ->
        IO.puts("   ❌ Benchmark failed: #{inspect(reason)}")
    end
  end
end

# Run the test
SyllableTest.run()
