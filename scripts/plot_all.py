"""Backward-compatible entry point for publication figure generation."""

try:
    from scripts.plot_results import main
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from plot_results import main


if __name__ == "__main__":
    main()
