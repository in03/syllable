defmodule SyllableElixir.Model do
  @moduledoc """
  Axon model for syllable counting.

  Replicates the TensorFlow model:
  - Bidirectional GRU (16 units, return_sequences=true)
  - GRU (16 units, return_sequences=false)
  - Dense (1 unit)
  """

  # Note: import Axon removed as we use explicit module calls
  require Logger

  @doc """
  Build the syllable counting model architecture.

  Input shape: {batch_size, 18, 28} (max_word_len, num_chars)
  Output shape: {batch_size, 1}
  """
            def build_model do
    # Input layer: {batch_size, 18, 28}
    input = Axon.input("input", shape: {nil, 18, 28})

        # Bidirectional GRU layer: 16 units, return sequences
    bidirectional_output =
      input
      |> Axon.bidirectional(
           &Axon.gru(&1, 16, return_sequences: true),
           &Axon.concatenate/1
         )
    # Output: {batch_size, 18, 32} (16 forward + 16 backward concatenated)

    # Second GRU layer: 16 units, return last output only
    gru_output =
      bidirectional_output
      |> Axon.gru(16)
    # Output: {batch_size, 16}

    # Dense output layer: 1 unit (syllable count)
    gru_output
    |> Axon.dense(1)
    # Output: {batch_size, 1}
  end

  @doc """
  Load model weights from the exported JSON file.
  """
  def load_weights(weights_json_path) do
    Logger.info("Loading model weights from #{weights_json_path}")

    weights_json_path
    |> File.read!()
    |> Jason.decode!()
    |> extract_layer_weights()
  end

  defp extract_layer_weights(weights_data) do
    Logger.info("Extracting weights for layers...")

    weights = weights_data["weights"]

    # Extract weights for each layer based on the actual JSON structure
    %{
      "bidirectional" => extract_bidirectional_weights(weights["bidirectional"]),
      "gru_1" => extract_gru_weights(weights["gru_1"]),
      "dense" => extract_dense_weights(weights["dense"])
    }
  end

  @doc """
  Create a prediction function from the model and weights.
  """
    def create_predict_fn(model, weights) do
    Logger.info("Creating prediction function...")

    # For now, create a simple prediction function that uses the model structure
    # We'll need to properly map the TensorFlow weights to Axon parameters
    # This is a simplified version that may need refinement

    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    # Use random initialization for now - in production you'd load the actual weights
    template_input = Nx.template({1, 18, 28}, :f32)
    params = init_fn.(template_input, %{})

    Logger.warning("Using random weights - need to properly map TensorFlow weights!")

    fn input ->
      predict_fn.(params, input)
    end
  end

  defp extract_bidirectional_weights(layer_data) do
    # The bidirectional layer contains forward and backward weights
    fwd_weights = layer_data["forward_weights"]
    bwd_weights = layer_data["backward_weights"]

    %{
      forward: %{
        kernel: Nx.tensor(fwd_weights[0]),
        recurrent_kernel: Nx.tensor(fwd_weights[1]),
        bias: Nx.tensor(fwd_weights[2])
      },
      backward: %{
        kernel: Nx.tensor(bwd_weights[0]),
        recurrent_kernel: Nx.tensor(bwd_weights[1]),
        bias: Nx.tensor(bwd_weights[2])
      }
    }
  end

  defp extract_gru_weights(layer_data) do
    weights = layer_data["weights"]

    %{
      kernel: Nx.tensor(weights[0]),
      recurrent_kernel: Nx.tensor(weights[1]),
      bias: Nx.tensor(weights[2])
    }
  end

  defp extract_dense_weights(layer_data) do
    weights = layer_data["weights"]

    %{
      kernel: Nx.tensor(weights[0]),
      bias: Nx.tensor(weights[1])
    }
  end

  defp map_weights_to_axon_params(our_weights, axon_params) do
    # This function maps our extracted weights to Axon's parameter structure
    # The exact mapping depends on how Axon names the parameters

    # For now, we'll create a simple mapping - this may need adjustment
    # based on the actual Axon parameter names when we test it
    Logger.debug("Mapping weights to Axon parameters...")
    Logger.debug("Axon parameter keys: #{inspect(Map.keys(axon_params))}")

    # Try to find the bidirectional layer parameters
    bidirectional_key =
      axon_params
      |> Map.keys()
      |> Enum.find(&String.contains?(to_string(&1), "bidirectional"))

    gru_key =
      axon_params
      |> Map.keys()
      |> Enum.find(&(String.contains?(to_string(&1), "gru") and not String.contains?(to_string(&1), "bidirectional")))

    dense_key =
      axon_params
      |> Map.keys()
      |> Enum.find(&String.contains?(to_string(&1), "dense"))

    mapped_params = %{}

    # Map bidirectional weights if found
    mapped_params =
      if bidirectional_key do
        Map.put(mapped_params, bidirectional_key, our_weights["bidirectional"])
      else
        mapped_params
      end

    # Map GRU weights if found
    mapped_params =
      if gru_key do
        Map.put(mapped_params, gru_key, our_weights["gru_1"])
      else
        mapped_params
      end

    # Map Dense weights if found
    mapped_params =
      if dense_key do
        Map.put(mapped_params, dense_key, our_weights["dense"])
      else
        mapped_params
      end

    Logger.debug("Mapped parameter keys: #{inspect(Map.keys(mapped_params))}")

    mapped_params
  end
end
