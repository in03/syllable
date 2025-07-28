defmodule SyllableElixir.Router do
  @moduledoc """
  HTTP API for the syllable counting service.

  Provides REST endpoints for syllable counting with JSON responses.
  """

  use Plug.Router

  alias SyllableElixir.Counter

  plug Plug.Logger
  plug :match
  plug Plug.Parsers, parsers: [:json], json_decoder: Jason
  plug :dispatch

  # Health check endpoint
  get "/health" do
    send_json(conn, 200, %{status: "ok", service: "syllable-counter"})
  end

  # Count syllables in a word or sentence
  post "/count" do
    case conn.body_params do
      %{"text" => text} when is_binary(text) ->
        handle_syllable_request(conn, text)

      _ ->
        send_json(conn, 400, %{
          error: "bad_request",
          message: "Request body must contain 'text' field"
        })
    end
  end

  # Count syllables via GET with query parameter
  get "/count" do
    case conn.query_params do
      %{"text" => text} when is_binary(text) ->
        handle_syllable_request(conn, text)

      _ ->
        send_json(conn, 400, %{
          error: "bad_request",
          message: "Query parameter 'text' is required"
        })
    end
  end

  # Get service statistics
  get "/stats" do
    case Counter.stats() do
      {:ok, stats} ->
        send_json(conn, 200, %{stats: stats})

      {:error, reason} ->
        send_json(conn, 500, %{
          error: "internal_error",
          message: "Failed to get stats: #{inspect(reason)}"
        })
    end
  end

  # API documentation
  get "/docs" do
    docs = %{
      service: "Syllable Counter API",
      version: "1.0.0",
      endpoints: [
        %{
          method: "GET",
          path: "/health",
          description: "Health check endpoint"
        },
        %{
          method: "POST",
          path: "/count",
          description: "Count syllables in text",
          body: %{text: "string"},
          example: %{text: "hello world"}
        },
        %{
          method: "GET",
          path: "/count?text=hello",
          description: "Count syllables via query parameter"
        },
        %{
          method: "GET",
          path: "/stats",
          description: "Get service performance statistics"
        }
      ],
      examples: [
        %{
          request: "POST /count",
          body: %{text: "hello"},
          response: %{
            text: "hello",
            syllable_count: 2,
            method: "dictionary",
            response_time_ms: 1.2
          }
        },
        %{
          request: "POST /count",
          body: %{text: "hello world"},
          response: %{
            text: "hello world",
            word_counts: [
              %{word: "hello", syllables: 2, method: "dictionary"},
              %{word: "world", syllables: 1, method: "dictionary"}
            ],
            total_syllables: 3,
            response_time_ms: 2.1
          }
        }
      ]
    }

    send_json(conn, 200, docs)
  end

  # Catch-all for unknown routes
  match _ do
    send_json(conn, 404, %{
      error: "not_found",
      message: "Endpoint not found. See /docs for available endpoints."
    })
  end

  # Private functions

  defp handle_syllable_request(conn, text) do
    start_time = System.monotonic_time(:millisecond)

    case Counter.count_syllables(text) do
      {:ok, count, method} when is_integer(count) ->
        # Single word response
        end_time = System.monotonic_time(:millisecond)
        response_time = end_time - start_time

        send_json(conn, 200, %{
          text: text,
          syllable_count: count,
          method: Atom.to_string(method),
          response_time_ms: response_time
        })

      {:ok, word_results} when is_list(word_results) ->
        # Multiple words response
        end_time = System.monotonic_time(:millisecond)
        response_time = end_time - start_time

        words = String.split(text, ~r/\s+/, trim: true)

        word_counts =
          Enum.zip(words, word_results)
          |> Enum.map(fn {word, result} ->
            case result do
              {:ok, count, method} ->
                %{word: word, syllables: count, method: Atom.to_string(method)}
              {:error, reason} ->
                %{word: word, error: Atom.to_string(reason)}
            end
          end)

        total_syllables =
          word_counts
          |> Enum.map(&Map.get(&1, :syllables, 0))
          |> Enum.sum()

        send_json(conn, 200, %{
          text: text,
          word_counts: word_counts,
          total_syllables: total_syllables,
          response_time_ms: response_time
        })

      {:error, reason} ->
        send_json(conn, 500, %{
          error: "processing_error",
          message: "Failed to count syllables: #{inspect(reason)}"
        })
    end
  end

  defp send_json(conn, status, data) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(status, Jason.encode!(data))
  end
end
