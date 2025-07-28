import Config

# Configure Nx to use binary backend for cross-platform compatibility
# EXLA is not available on Windows, so we use the BinaryBackend
config :nx, default_backend: Nx.BinaryBackend

# Syllable counter configuration
config :syllable_elixir,
  # Enable logging for debugging
  log_level: :info,

  # Optional: Enable web server
  enable_web_server: false,
  web_server_port: 4000
