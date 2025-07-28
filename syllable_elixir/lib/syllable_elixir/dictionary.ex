defmodule SyllableElixir.Dictionary do
  @moduledoc """
  CMU Pronunciation Dictionary-based syllable counting.

  Provides fast O(1) dictionary lookups for known words.
  """

  use GenServer

  @cmu_dict_path Application.app_dir(:syllable_elixir, "priv/cmudict/cmudict.dict")

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Look up syllable count for a word in the CMU dictionary.

  Returns `{:ok, count}` if found, `:error` if not found.
  """
  def lookup(word) when is_binary(word) do
    GenServer.call(__MODULE__, {:lookup, word})
  end

  @doc """
  Check if a word exists in the dictionary.
  """
  def has_word?(word) when is_binary(word) do
    case lookup(word) do
      {:ok, _count} -> true
      :error -> false
    end
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    dict = load_dictionary()
    {:ok, %{dict: dict, size: map_size(dict)}}
  end

  @impl true
  def handle_call({:lookup, word}, _from, %{dict: dict} = state) do
    # Clean word (remove punctuation, lowercase)
    clean_word =
      word
      |> String.downcase()
      |> String.trim_leading(~s{'"`})
      |> String.trim_trailing(~s{'"`.,!?;:})

    result = Map.get(dict, clean_word, :error)
    {:reply, result, state}
  end

  # Private functions

  defp load_dictionary do
    IO.puts("📚 Loading CMU pronunciation dictionary...")

    dict =
      @cmu_dict_path
      |> File.read!()
      |> String.split("\n", trim: true)
      |> Enum.reduce(%{}, &parse_dict_line/2)

    IO.puts("   Loaded #{map_size(dict)} words")
    dict
  end

  defp parse_dict_line(line, acc) do
    case String.split(line, " ", trim: true) do
      [word | phonemes] ->
        # Remove alternate pronunciation markers like "WORD(2)"
        clean_word =
          case String.split(word, "(") do
            [base_word | _] -> String.downcase(base_word)
            _ -> String.downcase(word)
          end

        # Count syllables by counting phonemes that end with digits
        syllable_count =
          phonemes
          |> Enum.count(&String.match?(&1, ~r/\d$/))

        # Take the first pronunciation if multiple exist
        Map.put_new(acc, clean_word, {:ok, syllable_count})

      _ ->
        acc
    end
  end
end
