defmodule SyllableElixir.CharEncoder do
  @moduledoc """
  Encodes strings to one-hot representation for the syllable counting model.

  This matches the CharacterEncoder from the Python implementation.
  """

  # Note: Nx.Defn import removed as it's not used

  # Characters from the trained model: ["'", "-", "a"-"z"]
  @chars ["'", "-"] ++ Enum.map(?a..?z, &<<&1>>)
  @char_to_index @chars |> Enum.with_index() |> Map.new()
  @num_chars length(@chars)
  @max_word_len 18

  def chars, do: @chars
  def num_chars, do: @num_chars
  def max_word_len, do: @max_word_len
  def char_to_index, do: @char_to_index

  @doc """
  Encodes a word into a one-hot tensor.

  Returns a tensor of shape {max_word_len, num_chars} where each row
  is a one-hot encoding of the character at that position.
  """
  def encode(word) when is_binary(word) do
    word
    |> String.downcase()
    |> String.graphemes()
    |> encode_chars()
  end

  defp encode_chars(chars) do
    # Create zero tensor
    tensor = Nx.broadcast(0.0, {@max_word_len, @num_chars})

    # Set one-hot values for each character
    chars
    |> Enum.take(@max_word_len)
    |> Enum.with_index()
    |> Enum.reduce(tensor, fn {char, pos}, acc ->
      case Map.get(@char_to_index, char) do
        nil -> acc  # Skip unknown characters
        char_idx -> Nx.put_slice(acc, [pos, char_idx], Nx.tensor([[1.0]]))
      end
    end)
  end

  @doc """
  Checks if a word can be encoded (all characters are known).
  """
  def encodable?(word) when is_binary(word) do
    word
    |> String.downcase()
    |> String.graphemes()
    |> Enum.all?(&Map.has_key?(@char_to_index, &1))
  end

  @doc """
  Batch encode multiple words.
  """
  def encode_batch(words) when is_list(words) do
    words
    |> Enum.map(&encode/1)
    |> Nx.stack()
  end
end
