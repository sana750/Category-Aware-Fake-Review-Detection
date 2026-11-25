import argparse

from transformer_model import train_transformer_model


def main():
    parser = argparse.ArgumentParser(
        description="Train or reuse the fine-tuned transformer model."
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--reuse",
        dest="force_retrain",
        action="store_false",
        default=False,
        help="Reuse saved transformer results when data/settings match (default)",
    )
    cache_group.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore saved transformer results and train again",
    )
    args = parser.parse_args()

    train_transformer_model(force_retrain=args.force_retrain)


if __name__ == "__main__":
    main()
