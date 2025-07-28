#!/usr/bin/env elixir

# Simple test to isolate the Axon model creation issue

Mix.install([
  {:axon, "~> 0.6"},
  {:nx, "~> 0.7"},
])

defmodule ModelTest do
  def test_simple_model do
    IO.puts("Testing simple Axon model creation...")

    # Test 1: Simple model that should work
    simple_model =
      Axon.input("input", shape: {nil, 18, 28})
      |> Axon.gru(16)
      |> Axon.dense(1)

    IO.puts("✅ Simple model created successfully")
    IO.inspect(simple_model)

    # Test 2: Try bidirectional without return_sequences first
    IO.puts("\nTesting bidirectional model...")

    try do
      bid_model =
        Axon.input("input", shape: {nil, 18, 28})
        |> Axon.bidirectional(
             fn x -> Axon.gru(x, 16) end,
             fn outputs -> Axon.concatenate(outputs) end
           )
        |> Axon.dense(1)

      IO.puts("✅ Bidirectional model created successfully")
      IO.inspect(bid_model)
    rescue
      error ->
        IO.puts("❌ Bidirectional model failed:")
        IO.inspect(error)
    end

    # Test 3: Try with return_sequences
    IO.puts("\nTesting bidirectional with return_sequences...")

    try do
      bid_seq_model =
        Axon.input("input", shape: {nil, 18, 28})
        |> Axon.bidirectional(
             fn x -> Axon.gru(x, 16, return_sequences: true) end,
             fn outputs -> Axon.concatenate(outputs) end
           )
        |> Axon.gru(16)
        |> Axon.dense(1)

      IO.puts("✅ Bidirectional with return_sequences created successfully")
      IO.inspect(bid_seq_model)
    rescue
      error ->
        IO.puts("❌ Bidirectional with return_sequences failed:")
        IO.inspect(error)
    end
  end
end

ModelTest.test_simple_model()
