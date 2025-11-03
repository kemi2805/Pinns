import h5py
import numpy as np

# Check h5py version and HDF5 version
print(f"h5py version: {h5py.__version__}")
print(f"HDF5 version: {h5py.version.hdf5_version}")
print(f"MPI enabled: {h5py.get_config().mpi}")

# Quick test write
with h5py.File('test.h5', 'w') as f:
    f.create_dataset('test', data=np.array([1, 2, 3]))
print("✓ h5py working correctly!")