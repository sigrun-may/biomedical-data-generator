# Copyright (c) 2025 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Tests for the command-line interface."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from biomedical_data_generator.__main__ import main


def test_main_with_config_file():
    """Test main function with valid config file."""
    # Create a minimal config YAML
    config_yaml = """
    n_informative: 5
    n_noise: 3
    class_configs:
      - n_samples: 25
      - n_samples: 25
    random_state: 42
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text(config_yaml)

        # Test without output file
        with patch("sys.argv", ["bdg", "--config", str(config_path)]):
            with patch("builtins.print") as mock_print:
                main()
                # Check that something was printed (the meta dict)
                assert mock_print.called
                # Verify it's valid JSON
                printed_arg = mock_print.call_args[0][0]
                meta_dict = json.loads(printed_arg)
                assert "samples_per_class" in meta_dict
                assert "n_classes" in meta_dict
                assert meta_dict["n_classes"] == 2


def test_main_with_output_file():
    """Test main function with output CSV file."""
    config_yaml = """
    n_informative: 3
    n_noise: 2
    class_configs:
      - n_samples: 10
      - n_samples: 10
    random_state: 123
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        output_path = Path(tmpdir) / "output.csv"
        config_path.write_text(config_yaml)

        with patch("sys.argv", ["bdg", "--config", str(config_path), "--out", str(output_path)]):
            with patch("builtins.print"):
                main()

        # Verify output file was created
        assert output_path.exists()

        # Verify the CSV content
        df = pd.read_csv(output_path)
        assert len(df) == 20
        assert "class" in df.columns
        # Should have 5 features (3 informative + 2 noise) + 1 class column
        assert len(df.columns) == 6


def test_main_missing_config():
    """Test main function with missing config argument."""
    with patch("sys.argv", ["bdg"]):
        with pytest.raises(SystemExit):
            main()


def test_main_invalid_config_file():
    """Test main function with non-existent config file."""
    with patch("sys.argv", ["bdg", "--config", "/nonexistent/config.yaml"]):
        with pytest.raises(FileNotFoundError):
            main()
