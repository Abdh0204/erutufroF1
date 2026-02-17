"""Phase 2 smoke tests -- verify data ingestion modules."""

from __future__ import annotations

import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch


class TestVerifyIdentifiers(unittest.TestCase):
    """Test identifier verification logic."""

    def test_successful_verification(self):
        from operator1.steps.verify_identifiers import verify_identifiers, VerifiedTarget

        mock_euler = MagicMock()
        mock_euler.get_profile.return_value = {
            "isin": "US0378331005",
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "country": "US",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "sub_industry": None,
            "currency": "USD",
            "exchange": "NASDAQ",
        }

        mock_fmp = MagicMock()
        mock_fmp.get_quote.return_value = {
            "name": "Apple Inc.",
            "price": 178.50,
            "symbol": "AAPL",
        }

        result = verify_identifiers("US0378331005", "AAPL", mock_euler, mock_fmp)

        self.assertIsInstance(result, VerifiedTarget)
        self.assertEqual(result.isin, "US0378331005")
        self.assertEqual(result.country, "US")
        self.assertEqual(result.fmp_symbol, "AAPL")
        self.assertEqual(result.sector, "Technology")

    def test_invalid_isin_raises(self):
        from operator1.steps.verify_identifiers import verify_identifiers, VerificationError
        from operator1.clients.eulerpool import EulerportAPIError

        mock_euler = MagicMock()
        mock_euler.get_profile.side_effect = EulerportAPIError("/profile", "Not found")

        mock_fmp = MagicMock()

        with self.assertRaises(VerificationError) as ctx:
            verify_identifiers("INVALID", "AAPL", mock_euler, mock_fmp)
        self.assertIn("Eulerpool", str(ctx.exception))

    def test_invalid_fmp_symbol_raises(self):
        from operator1.steps.verify_identifiers import verify_identifiers, VerificationError
        from operator1.clients.fmp import FMPAPIError

        mock_euler = MagicMock()
        mock_euler.get_profile.return_value = {
            "isin": "US0378331005", "ticker": "AAPL", "name": "Apple",
            "country": "US", "sector": "Tech", "industry": "CE",
            "currency": "USD", "exchange": "NASDAQ",
        }

        mock_fmp = MagicMock()
        mock_fmp.get_quote.side_effect = FMPAPIError("/quote", "Invalid symbol")

        with self.assertRaises(VerificationError) as ctx:
            verify_identifiers("US0378331005", "INVALID", mock_euler, mock_fmp)
        self.assertIn("FMP", str(ctx.exception))

    def test_missing_country_raises(self):
        from operator1.steps.verify_identifiers import verify_identifiers, VerificationError

        mock_euler = MagicMock()
        mock_euler.get_profile.return_value = {
            "isin": "XX", "ticker": "XX", "name": "XX",
            "country": None,  # no country
            "sector": "", "industry": "", "currency": "", "exchange": "",
        }
        mock_fmp = MagicMock()

        with self.assertRaises(VerificationError):
            verify_identifiers("XX", "XX", mock_euler, mock_fmp)


class TestEntityDiscoveryScoring(unittest.TestCase):
    """Test the entity match scoring function."""

    def test_exact_ticker_match(self):
        from operator1.steps.entity_discovery import _score_match
        score = _score_match("AAPL", {
            "ticker": "AAPL", "name": "Apple Inc.", "country": "US", "sector": "technology",
        }, "US", "technology")
        # 40 (ticker) + name_sim + 15 (country) + 15 (sector) should be high
        self.assertGreaterEqual(score, 70)

    def test_no_match(self):
        from operator1.steps.entity_discovery import _score_match
        score = _score_match("Apple Inc.", {
            "ticker": "MSFT", "name": "Microsoft Corp", "country": "JP", "sector": "finance",
        }, "US", "technology")
        self.assertLess(score, 70)

    def test_name_similarity(self):
        from operator1.steps.entity_discovery import _score_match
        # Query is the company name, not ticker, so ticker-exact match (40 pts) won't fire.
        # Name similarity (30) + country (15) + sector (15) = 60.
        score = _score_match("Apple Inc.", {
            "ticker": "AAPL", "name": "Apple Inc.", "country": "US", "sector": "technology",
        }, "US", "technology")
        self.assertGreaterEqual(score, 60)


class TestEntityDiscoveryFunctions(unittest.TestCase):
    """Test entity discovery helper functions."""

    def test_get_all_linked_isins(self):
        from operator1.steps.entity_discovery import (
            DiscoveryResult, LinkedEntity, get_all_linked_isins,
        )
        result = DiscoveryResult(linked={
            "competitors": [
                LinkedEntity("ISIN1", "T1", "N1", "US", "Tech", "competitors", 80),
                LinkedEntity("ISIN2", "T2", "N2", "US", "Tech", "competitors", 75),
            ],
            "suppliers": [
                LinkedEntity("ISIN2", "T2", "N2", "US", "Tech", "suppliers", 90),  # duplicate
                LinkedEntity("ISIN3", "T3", "N3", "DE", "Ind", "suppliers", 85),
            ],
        })
        isins = get_all_linked_isins(result)
        self.assertEqual(isins, ["ISIN1", "ISIN2", "ISIN3"])

    def test_empty_discovery(self):
        from operator1.steps.entity_discovery import DiscoveryResult, get_all_linked_isins
        result = DiscoveryResult()
        self.assertEqual(get_all_linked_isins(result), [])


class TestDataExtractionImports(unittest.TestCase):
    """Verify data extraction module imports cleanly."""

    def test_imports(self):
        from operator1.steps.data_extraction import (
            extract_all_data, EntityData, ExtractionResult, save_extraction_metadata,
        )
        self.assertTrue(callable(extract_all_data))

    def test_entity_data_defaults(self):
        from operator1.steps.data_extraction import EntityData
        e = EntityData(isin="TEST")
        self.assertEqual(e.isin, "TEST")
        self.assertTrue(e.quotes.empty)
        self.assertEqual(e.peers, [])


class TestMacroMappingImports(unittest.TestCase):
    """Verify macro mapping module imports cleanly."""

    def test_imports(self):
        from operator1.steps.macro_mapping import (
            fetch_macro_data, MacroDataset, save_macro_metadata,
        )
        self.assertTrue(callable(fetch_macro_data))

    def test_macro_dataset_defaults(self):
        from operator1.steps.macro_mapping import MacroDataset
        ds = MacroDataset(country_iso3="USA")
        self.assertEqual(ds.country_iso3, "USA")
        self.assertEqual(ds.indicators, {})
        self.assertEqual(ds.missing, [])


if __name__ == "__main__":
    unittest.main()
