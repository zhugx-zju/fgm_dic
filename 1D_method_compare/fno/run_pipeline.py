"""
Quick start script for FNO training pipeline
Runs: data generation -> training -> evaluation
"""
import os
import sys


def run_command(cmd, desc):
    """Run command with description"""
    print("\n" + "="*80)
    print(desc)
    print("="*80)
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error: {desc} failed with return code {ret}")
        sys.exit(1)


def main():
    print("="*80)
    print("FNO Training Pipeline - Quick Start")
    print("="*80)

    # Step 1: Generate data
    if not os.path.exists('train_data.pt'):
        run_command('python data_gen.py', 'Step 1: Generating training data')
    else:
        print("\nStep 1: Training data already exists, skipping...")

    # Step 2: Train model
    run_command('python train.py', 'Step 2: Training FNO model')

    # Step 3: Evaluate
    run_command('python eval.py', 'Step 3: Evaluating on test set')

    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)
    print("\nResults:")
    print("  - Model checkpoint: checkpoints/best_model.pt")
    print("  - Training logs: runs/ (view with tensorboard)")
    print("  - Evaluation plots: plots/")
    print("\nTo view training curves:")
    print("  tensorboard --logdir=runs")


if __name__ == '__main__':
    main()
