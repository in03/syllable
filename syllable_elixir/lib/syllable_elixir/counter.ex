defmodule SyllableElixir.Counter do
  @moduledoc """
  Main syllable counter that combines dictionary lookup with ML model fallback.

  Uses the same strategy as the Python implementation:
  1. Try CMU dictionary lookup first (fast)
  2. Fall back to ML model for unknown words
  """

  use GenServer
  require Logger

  alias SyllableElixir.{Dictionary, Model, CharEncoder}

  @weights_file Application.app_dir(:syllable_elixir, "priv/syllable_model_weights.json")

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Count syllables in a word or sentence.

  For single words, returns `{:ok, count}` or `{:error, reason}`.
  For sentences, returns `{:ok, [counts]}` where each count corresponds to a word.
  """
  def count_syllables(text) when is_binary(text) do
    GenServer.call(__MODULE__, {:count_syllables, text}, 10_000)
  end

  @doc """
  Count syllables for a single word (internal API).
  """
  def count_word(word) when is_binary(word) do
    GenServer.call(__MODULE__, {:count_word, word})
  end

  @doc """
  Get performance statistics.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    Logger.info("🧠 Initializing syllable counter...")

    # Start dictionary if not already started
    case GenServer.whereis(Dictionary) do
      nil -> Dictionary.start_link()
      _pid -> :ok
    end

    state = %{
      model: nil,
      predict_fn: nil,
      stats: %{
        total_requests: 0,
        dictionary_hits: 0,
        model_predictions: 0,
        errors: 0,
        avg_response_time_ms: 0.0
      },
      model_loaded: false
    }

    # Load model asynchronously
    send(self(), :load_model)

    {:ok, state}
  end

  @impl true
  def handle_info(:load_model, state) do
    case load_model() do
      {:ok, model, predict_fn} ->
        Logger.info("✅ Model loaded successfully")
        {:noreply, %{state | model: model, predict_fn: predict_fn, model_loaded: true}}

      {:error, reason} ->
        Logger.error("❌ Failed to load model: #{inspect(reason)}")
        {:noreply, %{state | model_loaded: false}}
    end
  end

  @impl true
  def handle_call({:count_syllables, text}, _from, state) do
    start_time = System.monotonic_time(:millisecond)

    result =
      text
      |> String.trim()
      |> split_into_words()
      |> count_words(state)

    end_time = System.monotonic_time(:millisecond)
    response_time = end_time - start_time

    new_state = update_stats(state, result, response_time)

    {:reply, result, new_state}
  end

  @impl true
  def handle_call({:count_word, word}, _from, state) do
    result = count_single_word(word, state)
    {:reply, result, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    {:reply, {:ok, state.stats}, state}
  end

  # Private functions

    defp load_model do
    try do
      Logger.info("🔄 Loading Axon model...")

      model = Model.build_model()

      # Check if weights file exists
      if File.exists?(@weights_file) do
        weights = Model.load_weights(@weights_file)
        predict_fn = Model.create_predict_fn(model, weights)
        {:ok, model, predict_fn}
      else
        Logger.warning("Weights file not found at #{@weights_file}, using simple model")
        # Create a simple predict function for testing
        predict_fn = fn _input -> Nx.tensor([[2.0]]) end  # Always predict 2 syllables
        {:ok, model, predict_fn}
      end
    rescue
      error ->
        Logger.error("Model loading failed: #{inspect(error)}")
        {:error, error}
    end
  end

  defp split_into_words(text) do
    text
    |> String.split(~r/\s+/, trim: true)
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
  end

  defp count_words(words, state) when is_list(words) do
    results =
      words
      |> Enum.map(&count_single_word(&1, state))

    # Return single count for single word, list for multiple words
    case results do
      [single_result] -> single_result
      multiple_results -> {:ok, multiple_results}
    end
  end

  defp count_single_word(word, state) do
    # Step 1: Try dictionary lookup
    case Dictionary.lookup(word) do
      {:ok, count} ->
        {:ok, count, :dictionary}

      :error ->
        # Step 2: Try model prediction
        predict_with_model(word, state)
    end
  end

  defp predict_with_model(_word, %{model_loaded: false}) do
    {:error, :model_not_loaded}
  end

  defp predict_with_model(word, %{predict_fn: predict_fn}) when is_function(predict_fn) do
    try do
      # Check if word can be encoded
      if CharEncoder.encodable?(word) and String.length(word) <= CharEncoder.max_word_len() do
        # Encode word to tensor
        input_tensor =
          word
          |> CharEncoder.encode()
          |> Nx.new_axis(0)  # Add batch dimension

        # Run prediction
        prediction = predict_fn.(input_tensor)

        # Extract and round the prediction
        syllable_count =
          prediction
          |> Nx.to_flat_list()
          |> List.first()
          |> round()
          |> max(1)  # Ensure at least 1 syllable

        {:ok, syllable_count, :model}
      else
        {:error, :word_not_encodable}
      end
    rescue
      error ->
        Logger.error("Model prediction error for word '#{word}': #{inspect(error)}")
        {:error, :prediction_failed}
    end
  end

  defp predict_with_model(_word, _state) do
    {:error, :model_not_available}
  end

  defp update_stats(state, result, response_time) do
    stats = state.stats
    new_total = stats.total_requests + 1

    # Update response time average
    new_avg_time =
      (stats.avg_response_time_ms * stats.total_requests + response_time) / new_total

    # Update hit/miss counters based on result
    {new_dict_hits, new_model_preds, new_errors} =
      case result do
        {:ok, _count, :dictionary} ->
          {stats.dictionary_hits + 1, stats.model_predictions, stats.errors}
        {:ok, _count, :model} ->
          {stats.dictionary_hits, stats.model_predictions + 1, stats.errors}
        {:ok, counts} when is_list(counts) ->
          # Multiple words - count each type
          {dict_hits, model_preds, errors} =
            Enum.reduce(counts, {0, 0, 0}, fn
              {:ok, _, :dictionary}, {d, m, e} -> {d + 1, m, e}
              {:ok, _, :model}, {d, m, e} -> {d, m + 1, e}
              {:error, _}, {d, m, e} -> {d, m, e + 1}
            end)

          {stats.dictionary_hits + dict_hits,
           stats.model_predictions + model_preds,
           stats.errors + errors}

        {:error, _} ->
          {stats.dictionary_hits, stats.model_predictions, stats.errors + 1}
      end

    new_stats = %{
      total_requests: new_total,
      dictionary_hits: new_dict_hits,
      model_predictions: new_model_preds,
      errors: new_errors,
      avg_response_time_ms: new_avg_time
    }

    %{state | stats: new_stats}
  end
end
