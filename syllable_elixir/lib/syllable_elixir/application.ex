defmodule SyllableElixir.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Start the dictionary first (required by counter)
      SyllableElixir.Dictionary,
      # Start the syllable counter GenServer
      SyllableElixir.Counter,
      # Optionally start the web server
      # {Plug.Cowboy, scheme: :http, plug: SyllableElixir.Router, options: [port: 4000]}
    ]

    opts = [strategy: :one_for_one, name: SyllableElixir.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
