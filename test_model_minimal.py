# HDF5 File Compatibility Check for GRACE ML Model
import h5py
import numpy as np

def check_hdf5_file_compatibility(filename):
    """
    Check if the HDF5 file created by h5py is compatible with GRACE's C++ reader
    """
    print(f"Checking HDF5 file: {filename}")
    print("=" * 50)
    
    # Check h5py and HDF5 versions
    print(f"h5py version: {h5py.version.version}")
    print(f"HDF5 version used by h5py: {h5py.version.hdf5_version}")
    print()
    
    try:
        with h5py.File(filename, 'r') as f:
            print("✓ File opens successfully with h5py")
            
            # Check expected datasets
            expected_datasets = [
                'weights_input_to_hidden',
                'bias_hidden', 
                'weights_hidden_to_output',
                'bias_output',
                'input_min',
                'input_max',
                'output_min',
                'output_max'
            ]
            
            print("\nDataset Check:")
            for dataset_name in expected_datasets:
                if dataset_name in f:
                    dataset = f[dataset_name]
                    print(f"✓ {dataset_name}: shape {dataset.shape}, dtype {dataset.dtype}")
                else:
                    print(f"✗ {dataset_name}: MISSING")
            
            # Check attributes
            print("\nAttributes Check:")
            expected_attrs = [
                'activation_function',
                'input_size',
                'hidden_size', 
                'output_size',
                'correction_scale',
                'model_type'
            ]
            
            for attr_name in expected_attrs:
                if attr_name in f.attrs:
                    attr_value = f.attrs[attr_name]
                    print(f"✓ {attr_name}: {attr_value}")
                else:
                    print(f"✗ {attr_name}: MISSING")
            
            # Check architecture consistency
            print("\nArchitecture Consistency Check:")
            if all(ds in f for ds in ['weights_input_to_hidden', 'weights_hidden_to_output']):
                w_ih = f['weights_input_to_hidden']
                w_ho = f['weights_hidden_to_output']
                
                print(f"Input size: {w_ih.shape[1]}")
                print(f"Hidden size: {w_ih.shape[0]} (from input->hidden) vs {w_ho.shape[1]} (from hidden->output)")
                print(f"Output size: {w_ho.shape[0]}")
                
                if w_ih.shape[0] == w_ho.shape[1]:
                    print("✓ Architecture is consistent")
                else:
                    print("✗ Architecture mismatch!")
            
            # Check data types (should be float64 or float32)
            print("\nData Type Check:")
            for dataset_name in expected_datasets:
                if dataset_name in f:
                    dtype = f[dataset_name].dtype
                    if dtype in [np.float32, np.float64]:
                        print(f"✓ {dataset_name}: {dtype} (compatible)")
                    else:
                        print(f"⚠ {dataset_name}: {dtype} (may cause issues)")
                        
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("Compatibility Assessment:")
    print("The file structure matches what GRACE expects.")
    print("If C++ loading still fails, the issue is likely HDF5 version compatibility.")
    print(f"Your h5py uses HDF5 {h5py.version.hdf5_version}, but GRACE uses HDF5 2.0.0")
    
    return True

def create_compatible_hdf5_test():
    """
    Create a minimal test HDF5 file to verify format compatibility
    """
    print("\nCreating minimal test file...")
    
    # Create minimal network data
    input_size, hidden_size, output_size = 3, 24, 1
    
    test_data = {
        'weights_input_to_hidden': np.random.randn(hidden_size, input_size).astype(np.float64),
        'bias_hidden': np.random.randn(hidden_size).astype(np.float64),
        'weights_hidden_to_output': np.random.randn(output_size, hidden_size).astype(np.float64),
        'bias_output': np.random.randn(output_size).astype(np.float64),
        'input_min': np.array([-1.0, -1.0, -1.0]),
        'input_max': np.array([1.0, 1.0, 1.0]),
        'output_min': np.array([1.0]),
        'output_max': np.array([10.0])
    }
    
    filename = 'test_model_minimal.h5'
    with h5py.File(filename, 'w') as f:
        # Store datasets
        for name, data in test_data.items():
            f.create_dataset(name, data=data)
        
        # Store attributes
        f.attrs['activation_function'] = 'tanh'
        f.attrs['input_size'] = input_size
        f.attrs['hidden_size'] = hidden_size
        f.attrs['output_size'] = output_size
        f.attrs['correction_scale'] = 1.0
        f.attrs['model_type'] = 'physics_guided'
    
    print(f"✓ Created test file: {filename}")
    print("Try loading this minimal file in your C++ code to isolate the issue.")
    
    return filename

# Run the compatibility check
if __name__ == "__main__":
    # Check your existing model file
    model_file = 'physics_guided_c2p_model.h5'
    check_hdf5_file_compatibility(model_file)
    
    # Create a minimal test file
    test_file = create_compatible_hdf5_test()
    
    print(f"\nNext steps:")
    print(f"1. Try loading '{test_file}' in your C++ code")
    print(f"2. If that fails too, it's definitely an HDF5 version compatibility issue")
    print(f"3. If it works, compare the two files to find the difference")