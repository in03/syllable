#!/usr/bin/env elixir

# Simple test for the Elixir syllable counter
# This focuses on dictionary lookup first, then tests model fallback

Mix.install([
  {:syllable_elixir, path: "."},
])

defmodule SimpleTest do
  def run do
    IO.puts("🧪 Simple Syllable Counter Test")
    IO.puts(String.duplicate("=", 40))

    # Start the application
    IO.puts("🔄 Starting application...")
    case Application.ensure_all_started(:syllable_elixir) do
      {:ok, _apps} ->
        IO.puts("✅ Application started")
      {:error, reason} ->
        IO.puts("❌ Failed to start: #{inspect(reason)}")
        exit(1)
    end

    # Give services time to start
    :timer.sleep(1000)

    # Test 1: Check if services are running
    test_services()

    # Test 2: Dictionary tests
    test_dictionary()

    # Test 3: Simple API tests
    test_api()

    IO.puts("\n🎉 Basic tests completed!")
  end

  defp test_services do
    IO.puts("\n🔍 Testing services...")

    case GenServer.whereis(SyllableElixir.Dictionary) do
      nil -> IO.puts("   ❌ Dictionary service not running")
      _pid -> IO.puts("   ✅ Dictionary service running")
    end

    case GenServer.whereis(SyllableElixir.Counter) do
      nil -> IO.puts("   ❌ Counter service not running")
      _pid -> IO.puts("   ✅ Counter service running")
    end
  end

  defp test_dictionary do
    IO.puts("\n📚 Testing dictionary lookups...")

    test_words = [
      "hello",
      "world",
      "elixir",
      "programming",
      "test"
    ]

    Enum.each(test_words, fn word ->
      case SyllableElixir.Dictionary.lookup(word) do
        {:ok, count} ->
          IO.puts("   ✅ #{word}: #{count} syllables")
        :error ->
          IO.puts("   ❌ #{word}: not found in dictionary")
      end
    end)
  end

  defp test_api do
    IO.puts("\n🎯 Testing main API...")

    # Test basic counting
    test_word("hello")
    test_word("world")
    test_word("elixir")
    test_word("programming")

    # Test a made-up word (should fall back to model or pattern)
    test_word("xyzabc")

    # Test sentence
    test_sentence("hello world")
  end

  defp test_word(word) do
    case SyllableElixir.count_detailed(word) do
      {:ok, count, method} ->
        IO.puts("   ✅ #{word}: #{count} syllables (#{method})")
      {:error, reason} ->
        IO.puts("   ❌ #{word}: #{inspect(reason)}")
    end
  end

  defp test_sentence(text) do
    IO.puts("\n📝 Testing sentence: \"#{text}\"")

    case SyllableElixir.count_detailed(text) do
      {:ok, word_details} when is_list(word_details) ->
        total = word_details
                |> Enum.map(&Map.get(&1, :syllables, 0))
                |> Enum.sum()

        IO.puts("   📊 Total syllables: #{total}")

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
end

# Run the test
SimpleTest.run()
