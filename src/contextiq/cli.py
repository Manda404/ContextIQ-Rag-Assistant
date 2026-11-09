"""Command-line entry point for ContextIQ RAG Assistant."""

import argparse

from contextiq.logging import log_operation


@log_operation
def main() -> None:
    parser = argparse.ArgumentParser(description="ContextIQ RAG Assistant")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
