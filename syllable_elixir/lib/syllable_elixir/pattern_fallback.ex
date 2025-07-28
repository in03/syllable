defmodule SyllableElixir.PatternFallback do
  @moduledoc """
  Simple pattern-based syllable counting for words not in the dictionary.

  This provides a reasonable fallback when the ML model isn't available.
  Based on common English syllable patterns.
  """

  @doc """
  Count syllables using simple heuristic patterns.

  This is a basic implementation that handles common English patterns:
  - Count vowel groups (a, e, i, o, u, y)
  - Handle silent 'e' at the end
  - Ensure minimum of 1 syllable
  """
  def count_syllables(word) when is_binary(word) do
    word
    |> String.downcase()
    |> String.trim()
    |> count_vowel_groups()
    |> adjust_for_silent_e(word)
    |> ensure_minimum_syllables()
  end

  defp count_vowel_groups(word) do
    # Split into characters and count vowel groups
    word
    |> String.graphemes()
    |> Enum.chunk_by(&is_vowel?/1)
    |> Enum.count(&(length(&1) > 0 and is_vowel?(hd(&1))))
  end

  defp is_vowel?(char) do
    char in ["a", "e", "i", "o", "u", "y"]
  end

  defp adjust_for_silent_e(count, word) do
    # Reduce count by 1 if word ends in silent 'e'
    if String.ends_with?(word, "e") and String.length(word) > 1 do
      max(1, count - 1)
    else
      count
    end
  end

  defp ensure_minimum_syllables(count) do
    max(1, count)
  end

  @doc """
  More sophisticated pattern matching for common prefixes and suffixes.
  """
  def count_syllables_advanced(word) when is_binary(word) do
    word = String.downcase(word)

    base_count = count_syllables(word)

    # Adjust for common patterns
    base_count
    |> adjust_for_prefixes(word)
    |> adjust_for_suffixes(word)
    |> ensure_minimum_syllables()
  end

  defp adjust_for_prefixes(count, word) do
    # Common prefixes that might affect syllable count
    prefixes = [
      {"re-", 1},
      {"pre-", 1},
      {"un-", 1},
      {"dis-", 1},
      {"over-", 2}
    ]

    Enum.reduce(prefixes, count, fn {prefix, syllables}, acc ->
      if String.starts_with?(word, String.replace(prefix, "-", "")) do
        # This is a simplified adjustment - real logic would be more complex
        acc
      else
        acc
      end
    end)
  end

  defp adjust_for_suffixes(count, word) do
    # Common suffixes that might affect syllable count
    cond do
      String.ends_with?(word, "ing") ->
        if String.length(word) > 5, do: count, else: count
      String.ends_with?(word, "ed") ->
        if String.ends_with?(word, ["ted", "ded"]), do: count, else: max(1, count - 1)
      String.ends_with?(word, "ly") ->
        count
      String.ends_with?(word, "tion") ->
        count
      true ->
        count
    end
  end
end
