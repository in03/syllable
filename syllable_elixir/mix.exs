defmodule SyllableElixir.MixProject do
  use Mix.Project

  def project do
    [
      app: :syllable_elixir,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {SyllableElixir.Application, []}
    ]
  end

      defp deps do
    [
      # Machine learning
      {:axon, "~> 0.6"},
      {:nx, "~> 0.7"},
      # Note: EXLA not included for Windows compatibility
      # On Unix systems, you can add: {:exla, "~> 0.7"} for better performance

      # JSON handling
      {:jason, "~> 1.4"},

      # HTTP server (optional - for microservice)
      {:plug_cowboy, "~> 2.6"},
      {:plug, "~> 1.14"},

      # Testing
      {:ex_doc, "~> 0.31", only: :dev, runtime: false}
    ]
  end
end
