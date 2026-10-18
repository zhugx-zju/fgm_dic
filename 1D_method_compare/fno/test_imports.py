"""
Simple test to check if imports work
"""
try:
    print("Testing imports...")

    print("1. Importing FEM1DBar...")
    from fem1d import FEM1DBar
    print("   ✓ FEM1DBar imported")

    print("2. Importing data_gen functions...")
    from data_gen import load_dataset, save_dataset
    print("   ✓ data_gen functions imported")

    print("3. Importing dataset functions...")
    from dataset import FNODataset, create_dataloaders
    print("   ✓ dataset functions imported")

    print("4. Importing model...")
    from model import FNOModel
    print("   ✓ FNOModel imported")

    print("\n✓ All imports successful!")

except Exception as e:
    print(f"\n✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
