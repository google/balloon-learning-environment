# Placeholder data

Used for unit testing the dataset. The `array.zarr` file was generated by
running the following command in this directory:

```bash
python3 -c "\
import zarr; import numpy as np; \
random_state = np.random.RandomState(seed=42); \
shape = (4, 21, 21, 10, 9, 2); \
chunks = (None,) * 5; \
values = np.asarray([-1.0, 1.0], dtype=np.float32); \
z = zarr.open('array.zarr', mode='w', shape=shape, chunks=chunks, dtype=np.float32); \
z[:] = random_state.choice(values, size=shape)"
```