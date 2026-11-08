import os
import h5py
import numpy as np
from scipy.sparse import eye, diags
from scipy.sparse.linalg import spsolve
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm


def whittaker_smooth(y, lmbd=1, d=2):
    """Smooth one time series while retaining its original NaN positions."""
    if len(y) == 0:
        return y

    valid_mask = ~np.isnan(y)
    y_clean = y[valid_mask]

    if len(y_clean) < 3:
        return y

    m = len(y_clean)
    E = eye(m, format="csr")
    if d == 2:
        D = diags([1, -2, 1], [0, 1, 2], shape=(m - 2, m), format="csr")
    elif d == 1:
        D = diags([1, -1], [0, 1], shape=(m - 1, m), format="csr")
    else:
        raise ValueError("Only difference orders d=1 and d=2 are supported")

    A = E + lmbd * (D.T @ D)
    y_smooth_clean = spsolve(A, y_clean)

    y_smooth = y.copy()
    y_smooth[valid_mask] = y_smooth_clean
    return y_smooth


def process_pixel_time_series(pixel_values, lmbd=1):
    """Apply second-order Whittaker smoothing to a pixel time series."""
    return whittaker_smooth(pixel_values, lmbd=lmbd, d=2)


def apply_whittaker_to_merged_h5(merged_h5_path, output_dir, lmbd=1, n_jobs=-1):
    """Smooth all pixel time series in a merged HDF5 file and save the result."""
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(merged_h5_path)
    name_no_ext = os.path.splitext(base_name)[0]
    output_file = os.path.join(output_dir, f"{name_no_ext}_whittaker.h5")

    print(f"Reading input file: {merged_h5_path}")
    with h5py.File(merged_h5_path, "r") as f:
        lon = f["longitude"][:]
        lat = f["latitude"][:]
        value_data = f["value"][:]

        timestamps = f.attrs["timestamps"]
        if isinstance(timestamps, np.ndarray):
            timestamps = [ts.decode("utf-8") if isinstance(ts, bytes) else str(ts) for ts in timestamps]
        else:
            timestamps = [str(timestamps)]

        attrs_to_copy = {key: f.attrs[key] for key in f.attrs.keys()}
        attrs_to_copy["processing"] = "Whittaker smoothing applied"
        attrs_to_copy["lambda"] = lmbd
        attrs_to_copy["smooth_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    n_pixels, n_times = value_data.shape
    print(f"Data dimensions: {n_pixels:,} pixels, {n_times} time points")
    print(f"Time span: {timestamps[0]} to {timestamps[-1]}")
    print(f"CPU workers: {n_jobs}")

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_pixel_time_series)(value_data[i, :], lmbd=lmbd)
        for i in tqdm(range(n_pixels), desc="Whittaker smoothing", total=n_pixels)
    )

    smoothed_value_data = np.array(results)
    print("Smoothing complete; writing output file...")

    with h5py.File(output_file, "w") as f:
        f.create_dataset("longitude", data=lon, compression="gzip")
        f.create_dataset("latitude", data=lat, compression="gzip")
        f.create_dataset("value", data=smoothed_value_data, compression="gzip")

        for key, val in attrs_to_copy.items():
            f.attrs[key] = val

    print(f"Processing complete. Output file: {output_file}")


if __name__ == "__main__":
    merged_h5_path = r"<PATH_TO_MERGED_HDF5_INPUT>"
    output_dir = r"<PATH_TO_OUTPUT_DIRECTORY>"

    lambda_value = 1
    n_jobs = -1

    apply_whittaker_to_merged_h5(
        merged_h5_path=merged_h5_path,
        output_dir=output_dir,
        lmbd=lambda_value,
        n_jobs=n_jobs,
    )

