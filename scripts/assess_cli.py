"""Command-line demo: python scripts/assess_cli.py <image> [<image> ...]"""
import sys

from meatqa.assess import assess_image, load_model


def main(paths: list[str]) -> None:
    """Assess each image and print the result, or the reason it was rejected.

    Args:
        paths: Image file paths.
    """
    bundle = load_model("models/svr_model.joblib")
    for p in paths:
        try:
            print(assess_image(p, bundle))
        except (FileNotFoundError, ValueError) as exc:
            print(f"REJECTED {p}: {exc}")


if __name__ == "__main__":
    main(sys.argv[1:])
