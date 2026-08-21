"""
src/cli.py
Unified Command Line Interface for Training, Evaluation, Inference, and System Exploration.
"""

import argparse
import sys
from pathlib import Path
from src.config import DEFAULT_MODEL_CONFIG, DEFAULT_PATHS, DEFAULT_TRAINING_CONFIG, ModelConfig, TrainingConfig
from src.evaluate import evaluate_model
from src.predict import DigitPredictor
from src.train import train_baseline, train_model
from src.utils import get_device_info, setup_logger
from src.visualization import generate_all_plots

logger = setup_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="mnist-cli",
        description="Robust MNIST Handwritten Digit Recognition Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    # Command: train
    train_p = subparsers.add_parser("train", help="Train the CNN model on MNIST")
    train_p.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_CONFIG.epochs, help="Number of training epochs")
    train_p.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING_CONFIG.batch_size, help="Batch size")
    train_p.add_argument("--lr", type=float, default=DEFAULT_TRAINING_CONFIG.learning_rate, help="Learning rate")
    train_p.add_argument("--dropout", type=float, default=DEFAULT_MODEL_CONFIG.dropout_rate, help="Dropout rate")

    # Command: baseline
    baseline_p = subparsers.add_parser("baseline", help="Train baseline MLP model for comparison")
    baseline_p.add_argument("--epochs", type=int, default=8, help="Number of baseline training epochs")

    # Command: evaluate
    eval_p = subparsers.add_parser("evaluate", help="Evaluate trained model on test dataset")
    eval_p.add_argument("--model-path", type=str, default=str(DEFAULT_PATHS.model_save_path), help="Path to .keras model")

    # Command: predict
    pred_p = subparsers.add_parser("predict", help="Predict digit from an image file")
    pred_p.add_argument("image_path", type=str, help="Path to input image file (PNG/JPG)")
    pred_p.add_argument("--model-path", type=str, default=str(DEFAULT_PATHS.model_save_path), help="Path to .keras model")
    pred_p.add_argument("--top-k", type=int, default=3, help="Top K predictions to output")

    # Command: visualize
    subparsers.add_parser("visualize", help="Generate all publication-grade plots and error heatmaps")

    # Command: system-info
    subparsers.add_parser("system-info", help="Display environment and device hardware information")

    return parser


def main():
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        training_cfg = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        model_cfg = ModelConfig(dropout_rate=args.dropout)
        logger.info(f"Initiating training with {args.epochs} epochs, batch size {args.batch_size}...")
        train_model(model_config=model_cfg, training_config=training_cfg)
        logger.info("Training complete. Generating plots...")
        generate_all_plots()

    elif args.command == "baseline":
        logger.info(f"Initiating baseline MLP training with {args.epochs} epochs...")
        train_baseline(epochs=args.epochs)

    elif args.command == "evaluate":
        logger.info(f"Evaluating model at '{args.model_path}'...")
        res = evaluate_model(model_path=Path(args.model_path))
        print("\n" + "=" * 50)
        print(f"Test Accuracy : {res['test_accuracy'] * 100:.2f}%")
        print(f"Test Loss     : {res['test_loss']:.4f}")
        print(f"Macro F1-Score: {res['macro_f1']:.4f}")
        print(f"Misclassified : {res['total_misclassified']} / {res['test_samples']}")
        print("=" * 50 + "\n")

    elif args.command == "predict":
        img_path = Path(args.image_path)
        if not img_path.exists():
            logger.error(f"Image file does not exist: {img_path}")
            sys.exit(1)
        predictor = DigitPredictor(model_path=args.model_path)
        result = predictor.predict(img_path, top_k=args.top_k)
        print("\n" + "=" * 50)
        if result.get("is_multi_digit", False):
            print(f"RECOGNIZED NUMBER: {result['full_number']} ({len(result['digits'])} digits)")
            print(f"AVG CONFIDENCE   : {result['confidence_percent']}")
            print("-" * 50)
            print("Per-Digit Breakdown:")
            for d in result["digits"]:
                print(f"  Digit {d['predicted_digit']} (Confidence: {d['confidence_percent']}, BBox: {d['box']})")
        else:
            print(f"PREDICTED DIGIT: {result['predicted_digit']}")
            print(f"CONFIDENCE     : {result['confidence_percent']}")
            print(f"STATUS         : {'High Confidence' if result['is_confident'] else 'Low Confidence / Uncertain'}")
            print("-" * 50)
            print("Top Predictions:")
            for rank, item in enumerate(result['top_k'], 1):
                print(f"  {rank}. Digit {item['digit']} -> {item['confidence'] * 100:.2f}%")
        print("=" * 50 + "\n")

    elif args.command == "visualize":
        logger.info("Generating all visualization assets...")
        generate_all_plots()
        logger.info(f"Plots saved to: {DEFAULT_PATHS.plots_dir}")

    elif args.command == "system-info":
        info = get_device_info()
        print("\n--- System & Hardware Information ---")
        for k, v in info.items():
            print(f"{k.replace('_', ' ').capitalize()}: {v}")
        print("------------------------------------\n")


if __name__ == "__main__":
    main()
