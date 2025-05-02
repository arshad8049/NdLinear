import os
import torch
from torch.utils.data import Dataset
import scipy.io
import pandas as pd
import torch.nn.functional as F
import numpy as np
import zipfile

class CFRPDataset(Dataset):
    """
    PyTorch Dataset for Stanford/NASA composite fatigue data.
    Expects directory structure:
      data/layupX/CouponID_F/...
    where X is 1, 2, or 3.
    """

    def __init__(self, root_dir, layup="layup1", window_size=1024, transform=None):
            # Resolve the layup folder name case-insensitively
        candidates = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d.lower() == layup.lower()
        ]
        if not candidates:
            raise ValueError(f"No layup folder found for '{layup}' in {root_dir}")
        self.root_dir = os.path.join(root_dir, candidates[0])
        self.window_size = window_size
        self.transform = transform
        self.samples = []

        # Iterate over coupon folders
        for item in os.listdir(self.root_dir):
            coupon_dir = os.path.join(self.root_dir, item)
            if not os.path.isdir(coupon_dir):
                continue

            # Find any log file (.xlsx, .xls, or .csv)
            log_files = [
                f for f in os.listdir(coupon_dir)
                if f.lower().endswith((".xlsx", ".xls", ".csv"))
                   and not f.startswith("._")
            ]
            if not log_files:
                continue
            log_path = os.path.join(coupon_dir, log_files[0])

            # Read into DataFrame
            if log_path.lower().endswith(".csv"):
                log_df = pd.read_csv(log_path)
            else:
                try:
                    log_df = pd.read_excel(log_path, engine="openpyxl")
                except (zipfile.BadZipFile, ValueError):
                    log_df = pd.read_excel(log_path, engine="xlrd")
            pzt_folder = os.path.join(coupon_dir, "PZT-data")
            if not os.path.isdir(pzt_folder):
                continue

            # For each cycle, load all .mat signals and store with label
            for _, row in log_df.iterrows():
                # Extract cycle from row, try both 'cycle' and 'cycles'
                cycle_val = row.get("cycle") if "cycle" in row else row.get("cycles", None)

                for mat_file in os.listdir(pzt_folder):
                    if not mat_file.endswith(".mat"):
                        continue
                    mat_path = os.path.join(pzt_folder, mat_file)
                    mat_data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
                    # Extract the 'coupon' struct
                    if 'coupon' not in mat_data:
                        continue
                    c = mat_data['coupon']
                    # If no label from CSV/Excel, fallback to coupon.cycles
                    if cycle_val is None and hasattr(c, "cycles"):
                        try:
                            cycle_val = int(c.cycles) if np.isscalar(c.cycles) else int(c.cycles[0])
                        except Exception:
                            cycle_val = 0
                    # Access path_data entries
                    pd_array = c.path_data
                    # Collect all actuator signals across trajectories
                    signals = []
                    # pd_array could be a 2D array of structs
                    for entry in np.atleast_1d(pd_array).flatten():
                        if hasattr(entry, 'signal_actuator'):
                            sa = entry.signal_actuator
                            if sa is not None:
                                signals.append(sa.flatten())
                    if not signals:
                        continue
                    signal = np.concatenate(signals)
                    # Trim or pad below __getitem__ will handle window_size
                    self.samples.append({
                        "signal": signal,
                        "cycle": cycle_val
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = torch.tensor(sample["signal"], dtype=torch.float32)

        # Trim or pad to fixed window_size
        if x.numel() >= self.window_size:
            x = x[: self.window_size]
        else:
            pad_size = self.window_size - x.numel()
            x = F.pad(x, (0, pad_size))

        y = torch.tensor(sample["cycle"], dtype=torch.float32)

        if self.transform is not None:
            x = self.transform(x)

        return x, y