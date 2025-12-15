"""
Test all three feature types: mel, mfcc, combined
"""

from transformers import AutoTokenizer
from datamodule.dataset import get_speech_dataset


class DictConfig:
    """Wrapper to support both dict.get() and dict.attribute access"""
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)
        self._config = config_dict
    
    def get(self, key, default=None):
        return self._config.get(key, default)


def test_feature_type(input_type):
    """Test a specific feature type"""
    
    print("\n" + "="*60)
    print(f"Testing: {input_type.upper()}")
    print("="*60)
    
    # Configuration
    config_dict = {
        'input_type': input_type,
        'sample_rate': 16000,
        'max_audio_length': 30.0,
        'mel_size': 80,
        'n_mfcc': 40,
        'n_fft': 400,
        'hop_length': 160,
        'win_length': 400,
        'fmin': 0,
        'fmax': 8000,
        'patch_length': 16,
        'patch_stride': 8,
        'normalize': False,
        'mel_input_norm': False,
        'mel_stats_path': None,
        'train_data_path': 'data/train_50h.jsonl',
        'val_data_path': 'data/dev_clean.jsonl',
        'test_data_path': 'data/test-clean.jsonl',
        'fix_length_audio': -1,
        'inference_mode': False,
        'prompt': None,
    }
    
    config = DictConfig(config_dict)
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"Loading test dataset...")
    dataset = get_speech_dataset(config, tokenizer, 'test')
    print(f"  Dataset loaded: {len(dataset)} samples")
    
    # Test single sample
    sample = dataset[0]
    print(f"\nSingle sample:")
    print(f"  audio_mel shape: {sample['audio_mel'].shape}")
    print(f"  audio_length: {sample['audio_length']}")
    
    # Test batch
    batch_size = 4
    samples = [dataset[i] for i in range(batch_size)]
    batch = dataset.collator(samples)
    
    print(f"\nBatch (size={batch_size}):")
    print(f"  audio_mel shape: {batch['audio_mel'].shape}")
    print(f"  audio_mel_post_mask shape: {batch['audio_mel_post_mask'].shape}")
    
    # Verify feature dimension
    feature_dim = batch['audio_mel'].shape[2]
    expected_dims = {'mel': 80, 'mfcc': 40, 'combined': 120}
    expected = expected_dims[input_type]
    
    print(f"\nFeature dimension check:")
    print(f"  Actual: {feature_dim}")
    print(f"  Expected: {expected}")
    
    if feature_dim == expected:
        print(f"  Status: PASSED")
        return True
    else:
        print(f"  Status: FAILED")
        return False


def main():
    """Test all three feature types"""
    
    print("="*60)
    print("Dataset Test Suite - All Feature Types")
    print("="*60)
    
    results = {}
    
    # Test all three types
    for feature_type in ['mel', 'mfcc', 'combined']:
        try:
            results[feature_type] = test_feature_type(feature_type)
        except Exception as e:
            print(f"\nTest FAILED for {feature_type}:")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results[feature_type] = False
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for feature_type, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {feature_type.upper():10s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("Overall: ALL TESTS PASSED")
        print("Dataset is ready for all three experiments")
    else:
        print("Overall: SOME TESTS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
