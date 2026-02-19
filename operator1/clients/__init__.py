"""API client modules for external data sources.

Available equity-data providers:
- ``eulerpool`` -- Eulerpool API (default if EULERPOOL_API_KEY is set)
- ``eod``       -- EOD Historical Data API (fallback if EOD_API_KEY is set)

Use :func:`equity_provider.create_equity_provider` to get the right
backend automatically based on available API keys.
"""
