"""
Test script to verify all combinations of use_fourier, is_train, and use_augs.
"""
import sys
sys.path.append('FastFlow')

from FastFlow.dataset import create_image_transform
from PIL import Image
import numpy as np

def test_pipeline_combination(use_fourier, is_train, use_augs, input_size=256):
    """Test a specific combination of flags."""
    print(f"\n{'='*60}")
    print(f"Testing: use_fourier={use_fourier}, is_train={is_train}, use_augs={use_augs}")
    print(f"{'='*60}")
    
    # Create transform
    transform = create_image_transform(input_size, use_fourier, is_train, use_augs)
    
    # Print pipeline
    print("\nPipeline steps:")
    for i, t in enumerate(transform.transforms):
        print(f"  {i+1}. {t.__class__.__name__}")
        if hasattr(t, 'transforms'):  # For Compose within Compose
            for j, sub_t in enumerate(t.transforms):
                print(f"     {i+1}.{j+1}. {sub_t.__class__.__name__}")
    
    # Test with a dummy image
    test_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    
    try:
        result = transform(test_img)
        print(f"\n✅ Success! Output shape: {result.shape}")
        print(f"   Output range: [{result.min():.3f}, {result.max():.3f}]")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
    
    return True

def main():
    """Test all combinations."""
    print("\n" + "="*60)
    print("TESTING ALL PIPELINE COMBINATIONS")
    print("="*60)
    
    combinations = [
        # (use_fourier, is_train, use_augs, description)
        (False, False, False, "Test RGB (no augs)"),
        (False, False, True,  "Test RGB (use_augs ignored in test)"),
        (False, True,  False, "Train RGB (no augs)"),
        (False, True,  True,  "Train RGB (with augs)"),
        (True,  False, False, "Test Fourier (no augs)"),
        (True,  False, True,  "Test Fourier (use_augs ignored in test)"),
        (True,  True,  False, "Train Fourier (no augs)"),
        (True,  True,  True,  "Train Fourier (with augs)"),
    ]
    
    for use_fourier, is_train, use_augs, desc in combinations:
        print(f"\n\n>>> {desc}")
        test_pipeline_combination(use_fourier, is_train, use_augs)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
