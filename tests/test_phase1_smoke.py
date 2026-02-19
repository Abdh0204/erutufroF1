"""Phase 1 smoke tests -- verify imports, configs, and basic module structure."""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch, MagicMock


class TestImports(unittest.TestCase):
    """Verify all Phase 1 modules import without error."""

    def test_package_import(self):
        import operator1
        self.assertTrue(hasattr(operator1, "__version__"))

    def test_constants_import(self):
        from operator1.constants import DATE_START, DATE_END, EPSILON
        self.assertIsInstance(DATE_START, date)
        self.assertIsInstance(DATE_END, date)
        self.assertGreater(DATE_END, DATE_START)
        self.assertGreater(EPSILON, 0)

    def test_config_loader_import(self):
        from operator1.config_loader import load_config, get_global_config
        self.assertTrue(callable(load_config))
        self.assertTrue(callable(get_global_config))

    def test_http_utils_import(self):
        from operator1.http_utils import (
            cached_get, inject_api_key, get_request_log, HTTPError,
        )
        self.assertTrue(callable(cached_get))
        self.assertTrue(callable(inject_api_key))

    def test_eulerpool_client_import(self):
        from operator1.clients.eulerpool import EulerportClient, EulerportAPIError
        self.assertTrue(callable(EulerportClient))

    def test_eod_client_import(self):
        from operator1.clients.eod import EODClient, EODAPIError
        self.assertTrue(callable(EODClient))

    def test_equity_provider_import(self):
        from operator1.clients.equity_provider import EquityProvider, create_equity_provider
        self.assertTrue(callable(create_equity_provider))

    def test_fmp_client_import(self):
        from operator1.clients.fmp import FMPClient, FMPAPIError
        self.assertTrue(callable(FMPClient))

    def test_world_bank_client_import(self):
        from operator1.clients.world_bank import WorldBankClient, WorldBankAPIError
        self.assertTrue(callable(WorldBankClient))

    def test_gemini_client_import(self):
        from operator1.clients.gemini import GeminiClient
        self.assertTrue(callable(GeminiClient))

    def test_subpackage_imports(self):
        import operator1.steps
        import operator1.features
        import operator1.analysis
        import operator1.quality
        import operator1.estimation
        import operator1.models
        import operator1.report


class TestConfigs(unittest.TestCase):
    """Verify YAML config files parse correctly."""

    def test_global_config(self):
        from operator1.config_loader import load_config
        cfg = load_config("global_config")
        self.assertIn("timeout_s", cfg)
        self.assertIn("max_retries", cfg)
        self.assertIn("FORCE_REBUILD", cfg)

    def test_world_bank_indicator_map(self):
        from operator1.config_loader import load_config
        cfg = load_config("world_bank_indicator_map")
        self.assertIn("inflation_rate_yoy", cfg)
        self.assertIn("gdp_growth", cfg)
        self.assertEqual(cfg["inflation_rate_yoy"], "FP.CPI.TOTL.ZG")

    def test_country_protection_rules(self):
        from operator1.config_loader import load_config
        cfg = load_config("country_protection_rules")
        self.assertIn("strategic_sectors", cfg)
        self.assertIsInstance(cfg["strategic_sectors"], list)
        self.assertIn("defense", cfg["strategic_sectors"])

    def test_survival_hierarchy(self):
        from operator1.config_loader import load_config
        cfg = load_config("survival_hierarchy")
        self.assertIn("tiers", cfg)
        self.assertIn("regimes", cfg)
        self.assertIn("normal", cfg["regimes"])
        # Verify weights sum to 1.0
        for regime_name, regime in cfg["regimes"].items():
            total = sum(regime["weights"])
            self.assertAlmostEqual(total, 1.0, places=5,
                                   msg=f"Weights for {regime_name} sum to {total}")

    def test_missing_config_raises(self):
        from operator1.config_loader import load_config
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent_config")


class TestAPIKeyInjection(unittest.TestCase):
    """Verify the URL API key injection logic."""

    def test_inject_to_url_without_params(self):
        from operator1.http_utils import inject_api_key
        result = inject_api_key("https://api.example.com/data", "MYKEY")
        self.assertEqual(result, "https://api.example.com/data?apikey=MYKEY")

    def test_inject_to_url_with_params(self):
        from operator1.http_utils import inject_api_key
        result = inject_api_key("https://api.example.com/data?symbol=AAPL", "MYKEY")
        self.assertEqual(result, "https://api.example.com/data?symbol=AAPL&apikey=MYKEY")


class TestSecretsLoader(unittest.TestCase):
    """Verify secrets loading logic."""

    @patch.dict("os.environ", {
        "EULERPOOL_API_KEY": "test_euler",
        "FMP_API_KEY": "test_fmp",
        "GEMINI_API_KEY": "test_gemini",
    })
    def test_load_from_env(self):
        from operator1.secrets_loader import load_secrets
        secrets = load_secrets()
        self.assertEqual(secrets["EULERPOOL_API_KEY"], "test_euler")
        self.assertEqual(secrets["FMP_API_KEY"], "test_fmp")
        self.assertEqual(secrets["GEMINI_API_KEY"], "test_gemini")

    @patch.dict("os.environ", {
        "EOD_API_KEY": "test_eod",
        "FMP_API_KEY": "test_fmp",
        "GEMINI_API_KEY": "test_gemini",
    })
    def test_load_eod_from_env(self):
        """EOD_API_KEY should be accepted as alternative to EULERPOOL_API_KEY."""
        from operator1.secrets_loader import load_secrets
        secrets = load_secrets()
        self.assertEqual(secrets["EOD_API_KEY"], "test_eod")
        self.assertEqual(secrets["FMP_API_KEY"], "test_fmp")
        self.assertNotIn("EULERPOOL_API_KEY", secrets)

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_keys_raises(self):
        from operator1.secrets_loader import load_secrets
        with self.assertRaises(SystemExit):
            load_secrets()

    @patch.dict("os.environ", {
        "FMP_API_KEY": "test_fmp",
        "GEMINI_API_KEY": "test_gemini",
    }, clear=True)
    def test_missing_equity_key_raises(self):
        """Pipeline should fail if neither EULERPOOL nor EOD key is set."""
        from operator1.secrets_loader import load_secrets
        with self.assertRaises(SystemExit):
            load_secrets()


class TestHTTPRetries(unittest.TestCase):
    """Verify retry and caching behaviour of cached_get."""

    @patch("operator1.http_utils.requests.get")
    def test_retry_on_500(self, mock_get):
        from operator1.http_utils import cached_get, clear_request_log, HTTPError

        clear_request_log()

        # First two calls return 500, third returns 200
        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.text = "Internal Server Error"
        resp_500.headers = {}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"data": "ok"}

        mock_get.side_effect = [resp_500, resp_500, resp_200]

        result = cached_get(
            "https://api.example.com/test",
            cache_dir="/tmp/test_cache_op1",
            ttl_hours=0.001,
        )
        self.assertEqual(result, {"data": "ok"})
        self.assertEqual(mock_get.call_count, 3)

    @patch("operator1.http_utils.requests.get")
    def test_all_retries_exhausted(self, mock_get):
        from operator1.http_utils import cached_get, clear_request_log, HTTPError

        clear_request_log()

        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.text = "fail"
        resp_500.headers = {}
        mock_get.return_value = resp_500

        with self.assertRaises(HTTPError):
            cached_get(
                "https://api.example.com/fail",
                cache_dir="/tmp/test_cache_op1",
                ttl_hours=0.001,
            )


if __name__ == "__main__":
    unittest.main()
