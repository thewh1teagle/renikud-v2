"""
Main entry point for Hebrew Nikud training.

This script serves as the entry point for training the Hebrew Nikud model.
All configuration is handled through config.py with argparse.
"""

from train_loop import main as train_main


def main():
    """Entry point for training."""
    train_main()


if __name__ == "__main__":
    main()

