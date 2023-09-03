from mpi4py import MPI
import h5py
import numpy as np

def parallel_write_to_hdf5(file_path, dataset_name, data, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    data_size = len(data)
    chunk_size = data_size // size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank < size - 1 else data_size

    with h5py.File(file_path, 'w', driver='mpio', comm=comm) as f:
        dset = f.create_dataset(dataset_name, shape=(data_size,), dtype=data.dtype)

        dset[start_idx:end_idx] = data[start_idx:end_idx]

if __name__ == '__main__':
    file_path = 'data_parallel.h5'
    dataset_name = 'my_dataset'
    total_data_size = 100000
    data = np.random.rand(total_data_size)  # Replace with your data

    comm = MPI.COMM_WORLD
    parallel_write_to_hdf5(file_path, dataset_name, data, comm)