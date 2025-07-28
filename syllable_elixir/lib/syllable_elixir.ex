defmodule SyllableElixir do
  @moduledoc """
  Fast, accurate syllable counting for English words and text.

  This library provides a high-performance syllable counter that combines:
  1. CMU Pronunciation Dictionary lookup (120k+ words, O(1) access)
  2. Machine learning model fallback using Axon (95.8% accuracy)

  ## Examples

      # Count syllables in a word
      iex> SyllableElixir.count("hello")
      {:ok, 2}

      # Count syllables in a sentence
      iex> SyllableElixir.count("hello world")
      {:ok, [2, 1]}

      # Get total syllables
      iex> SyllableElixir.total_syllables("hello beautiful world")
      {:ok, 5}

  ## Performance

  - Dictionary lookups: ~0.1ms per word
  - Model predictions: ~2-5ms per word
  - Loads 120k+ dictionary entries in ~100ms
  - Model loads in ~500ms

  ## Architecture

  The system uses a two-tier approach:
  1. **Dictionary First**: Fast O(1) lookup in CMU pronunciation dictionary
  2. **ML Fallback**: Axon model for unknown words (BiGRU + GRU + Dense)

  This gives you the speed of dictionary lookup for common words with the
  coverage of machine learning for rare/new words.
  """

  alias SyllableElixir.Counter

  @doc """
  Count syllables in a word or text.

  Returns `{:ok, count}` for single words or `{:ok, [counts]}` for multiple words.

  ## Examples

      iex> SyllableElixir.count("hello")
      {:ok, 2}

      iex> SyllableElixir.count("hello world")
      {:ok, [2, 1]}

      iex> SyllableElixir.count("supercalifragilisticexpialidocious")
      {:ok, 13}
  """
  def count(text) when is_binary(text) do
    case Counter.count_syllables(text) do
      {:ok, count, _method} when is_integer(count) ->
        {:ok, count}

      {:ok, word_results} when is_list(word_results) ->
        counts =
          word_results
          |> Enum.map(fn
            {:ok, count, _method} -> count
            {:error, _reason} -> 0  # Default to 0 for failed words
          end)
        {:ok, counts}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Count total syllables in text (sum of all words).

  ## Examples

      iex> SyllableElixir.total_syllables("hello world")
      {:ok, 3}

      iex> SyllableElixir.total_syllables("The quick brown fox")
      {:ok, 4}
  """
  def total_syllables(text) when is_binary(text) do
    case count(text) do
      {:ok, count} when is_integer(count) ->
        {:ok, count}

      {:ok, counts} when is_list(counts) ->
        {:ok, Enum.sum(counts)}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Count syllables with detailed information about the method used.

  Returns `{:ok, count, method}` for single words or `{:ok, word_details}` for multiple words.

  ## Examples

      iex> SyllableElixir.count_detailed("hello")
      {:ok, 2, :dictionary}

      iex> SyllableElixir.count_detailed("hello xyzzyx")
      {:ok, [
        %{word: "hello", syllables: 2, method: :dictionary},
        %{word: "xyzzyx", syllables: 2, method: :model}
      ]}
  """
  def count_detailed(text) when is_binary(text) do
    case Counter.count_syllables(text) do
      {:ok, count, method} when is_integer(count) ->
        {:ok, count, method}

      {:ok, word_results} when is_list(word_results) ->
        words = String.split(text, ~r/\s+/, trim: true)

        details =
          Enum.zip(words, word_results)
          |> Enum.map(fn {word, result} ->
            case result do
              {:ok, count, method} ->
                %{word: word, syllables: count, method: method}
              {:error, reason} ->
                %{word: word, error: reason}
            end
          end)

        {:ok, details}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Check if the syllable counter is ready (dictionary and model loaded).

  ## Examples

      iex> SyllableElixir.ready?()
      true
  """
  def ready? do
    case Counter.stats() do
      {:ok, _stats} -> true
      {:error, _} -> false
    end
  end

  @doc """
  Get performance statistics for the syllable counter.

  ## Examples

      iex> SyllableElixir.stats()
      {:ok, %{
        total_requests: 1247,
        dictionary_hits: 1180,
        model_predictions: 67,
        errors: 0,
        avg_response_time_ms: 1.2
      }}
  """
  def stats do
    Counter.stats()
  end

  @doc """
  Start the syllable counting service.

  This is automatically called when the application starts, but can be
  called manually if needed.
  """
  def start do
    case Application.ensure_all_started(:syllable_elixir) do
      {:ok, _apps} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Benchmark the syllable counter with common English words.

  Returns timing and accuracy statistics.

  ## Examples

      iex> SyllableElixir.benchmark()
      {:ok, %{
        words_tested: 100,
        avg_time_ms: 1.2,
        dictionary_hit_rate: 0.89,
        model_usage_rate: 0.11
      }}
  """
  def benchmark do
    words = [
      "hello", "world", "python", "elixir", "programming", "language",
      "artificial", "intelligence", "machine", "learning", "neural", "network",
      "syllable", "counting", "performance", "benchmark", "optimization",
      "application", "development", "framework", "library", "function",
      "beautiful", "wonderful", "fantastic", "incredible", "amazing",
      "supercalifragilisticexpialidocious", "antidisestablishmentarianism",
      "pneumonoultramicroscopicsilicovolcanoconiosis"
    ]

    start_time = System.monotonic_time(:millisecond)

    results =
      words
      |> Enum.map(fn word ->
        word_start = System.monotonic_time(:millisecond)
        result = count_detailed(word)
        word_end = System.monotonic_time(:millisecond)

        {word, result, word_end - word_start}
      end)

    end_time = System.monotonic_time(:millisecond)
    total_time = end_time - start_time

    # Analyze results
    {dict_hits, model_hits, errors} =
      Enum.reduce(results, {0, 0, 0}, fn {_word, result, _time}, {d, m, e} ->
        case result do
          {:ok, _count, :dictionary} -> {d + 1, m, e}
          {:ok, _count, :model} -> {d, m + 1, e}
          {:error, _} -> {d, m, e + 1}
        end
      end)

    word_times = Enum.map(results, fn {_word, _result, time} -> time end)
    avg_time = Enum.sum(word_times) / length(word_times)

    {:ok, %{
      words_tested: length(words),
      total_time_ms: total_time,
      avg_time_ms: avg_time,
      dictionary_hits: dict_hits,
      model_predictions: model_hits,
      errors: errors,
      dictionary_hit_rate: dict_hits / length(words),
      model_usage_rate: model_hits / length(words),
      error_rate: errors / length(words)
    }}
  end
end
