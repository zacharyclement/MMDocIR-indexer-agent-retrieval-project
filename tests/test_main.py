"""Tests for CLI parser behavior."""

from indexer.main import build_parser


def test_build_parser_parses_index_all() -> None:
    parser = build_parser()

    args = parser.parse_args(["index", "--all"])

    assert args.command == "index"
    assert args.all is True
    assert args.file is None


def test_build_parser_parses_validate_file() -> None:
    parser = build_parser()

    args = parser.parse_args(["validate", "--file", "watch_d.pdf"])

    assert args.command == "validate"
    assert args.all is False
    assert args.file == "watch_d.pdf"


def test_build_parser_parses_show_mapping_gaps() -> None:
    parser = build_parser()

    args = parser.parse_args(["show-mapping-gaps"])

    assert args.command == "show-mapping-gaps"
