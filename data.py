# ---------------------------
# SCRIPT: data.py (Modified)
# ---------------------------

# Standard library
import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

# Third-party libraries
import torch
import numpy as np
import h5py
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from tqdm import tqdm

# PyTorch
from torch.utils.data import Dataset, DataLoader, Sampler

# PyTorch Lightning
import lightning.pytorch as pl

# --- Helper functions for SALT model (used for energy loss in main script) ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def deep_update(base, updates):
    if isinstance(base, dict) and isinstance(updates, dict):
        for k, v in updates.items():
            base[k] = deep_update(base.get(k), v)
        return base
    return updates

def load_salt_model_from_config(base_config_path, user_config_path, ckpt_path, **kwargs):
    """
    Loads a pre-trained SALT model. This is used by the main training script
    to calculate the energy conservation loss.
    """
    from salt.modelwrapper import ModelWrapper
    with open(base_config_path) as f: base_config = yaml.safe_load(f)
    with open(user_config_path) as f: user_config = yaml.safe_load(f)
    config = deep_update(base_config, user_config)
    model = ModelWrapper.load_from_checkpoint(ckpt_path, config=config, **kwargs)
    return model.to(DEVICE).eval(), config

# --- 1. The High-Performance Sampler (Unchanged) ---
class OriginalLikeBatchSampler(Sampler):
    """
    A batch sampler that yields slices, allowing for direct, chunked reading
    from an HDF5 file. This is highly efficient as it avoids iterating
    over individual indices.
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.nonzero_last_batch = int(self.n_batches) < self.n_batches
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __len__(self):
        return int(self.n_batches) + int(not self.drop_last and self.nonzero_last_batch)

    def __iter__(self):
        if self.shuffle:
            batch_ids = torch.randperm(int(self.n_batches))
        else:
            batch_ids = torch.arange(int(self.n_batches))

        for batch_id in batch_ids:
            start, stop = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            yield np.s_[int(start):int(stop)]

        if not self.drop_last and self.nonzero_last_batch:
            start, stop = int(self.n_batches) * self.batch_size, self.dataset_length
            yield np.s_[int(start):int(stop)]

# --- 2. The High-Performance Constituent Dataset (MODIFIED) ---
class FastOriginalDataset(Dataset):
    """
    A PyTorch Dataset designed for fast, parallelized reading from a single HDF5 file.
    Each worker opens its own file handle, and it reads data in batches (slices).

    MODIFIED: This dataset now returns a tuple of two dictionaries:
    1. inputs: Contains the feature tensors with NaNs replaced by 0.0.
    2. masks: Contains boolean masks where True indicates a padded (originally NaN) value.
    """
    def __init__(self, file_path, variables: Dict[str, List[str]], input_map: Dict[str, str] = None,
                 constituent_name: str = "constituents"):
        super().__init__()
        self.file_path = file_path
        self.variables = variables
        self.constituent_name = constituent_name
        self.input_map = input_map if input_map is not None else {k: k for k in self.variables}

        with h5py.File(self.file_path, 'r') as f:
            self._len = len(f[next(iter(self.input_map.values()))])

        self.file, self.dsets, self.arrays = None, None, None

    def __len__(self):
        return self._len

    def _initialize_worker(self):
        """Initializes the HDF5 file handle and datasets for each worker process."""
        self.file = h5py.File(self.file_path, 'r')
        self.dsets = {name: self.file[dset_name] for name, dset_name in self.input_map.items()}
        self.arrays = {name: np.array([], dtype=dset.dtype) for name, dset in self.dsets.items()}

    def __getitem__(self, slice_obj: slice) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.file is None:
            self._initialize_worker()

        batch_size = slice_obj.stop - slice_obj.start
        inputs = {}
        masks = {} # NEW: Dictionary to store padding masks

        for name, features in self.variables.items():
            batch_array = self.arrays[name]
            new_shape = (batch_size,) + self.dsets[name].shape[1:]
            batch_array.resize(new_shape, refcheck=False)
            self.dsets[name].read_direct(batch_array, source_sel=slice_obj)

            unstructured_array = s2u(batch_array[features], dtype=np.float32)
            raw_tensor = torch.from_numpy(unstructured_array)

            # --- NEW: Generate mask from NaNs and then clean the tensor ---
            # The mask is True where values are NaN (i.e., padded).
            # This is the convention expected by PyTorch's attention mechanisms.
            masks[name] = torch.isnan(raw_tensor)

            # Replace NaNs with 0.0 for safe processing in the model.
            inputs[name] = torch.nan_to_num(raw_tensor, nan=0.0)

        # MODIFIED: Return both the cleaned inputs and the corresponding masks.
        return inputs, masks

# --- 3. The Paired Dataset for CycleGAN (MODIFIED) ---
class PairedFastDataset(Dataset):
    """
    A wrapper dataset that pairs two FastOriginalDataset instances for CycleGAN.
    It takes a slice and fetches the corresponding batch from both Domain A and Domain B.

    MODIFIED: This dataset now returns the cleaned data along with the padding masks
    for the constituents from each domain.
    """
    def __init__(self, file_path_A: str, file_path_B: str, variables: dict, input_map: dict, constituent_name: str):
        self.dset_A = FastOriginalDataset(file_path_A, variables, input_map, constituent_name)
        self.dset_B = FastOriginalDataset(file_path_B, variables, input_map, constituent_name)
        self.constituent_name = constituent_name
        self._len = min(len(self.dset_A), len(self.dset_B))

    def __len__(self):
        return self._len

    def __getitem__(self, slice_obj: slice):
        # MODIFIED: Unpack the new return signature (inputs, masks)
        batch_A, masks_A = self.dset_A[slice_obj]
        batch_B, masks_B = self.dset_B[slice_obj]

        jets_A = batch_A['jets']
        jets_B = batch_B['jets']
        constituents_A = batch_A[self.constituent_name]
        constituents_B = batch_B[self.constituent_name]

        # NEW: Extract the masks corresponding to the constituents
        # The mask will have shape (batch, n_constituents)
        constituent_mask_A = masks_A[self.constituent_name].any(dim=-1)
        constituent_mask_B = masks_B[self.constituent_name].any(dim=-1)

        # MODIFIED: Return the masks as additional items in the tuple
        return (jets_A, jets_B, constituents_A, constituents_B, constituent_mask_A, constituent_mask_B)

# --- 4. The Main High-Performance Lightning DataModule for CycleGAN (MODIFIED) ---
class TransformerCycleGANDataModule(pl.LightningDataModule):
    def __init__(
        self,
        mc_files: dict,
        data_files: dict,
        variables: dict,
        input_map: dict,
        constituent_name: str,
        batch_size: int = 2048,
        num_workers: int = 16,
        num_batches_for_stats: int = 100, # Batches to use for normalization stats
        **kwargs # Absorb unused hyperparameters
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str = None):
        common_args = {
            "variables": self.hparams.variables,
            "input_map": self.hparams.input_map,
            "constituent_name": self.hparams.constituent_name
        }
        if stage == "fit" or stage is None:
            self.train_dataset = PairedFastDataset(
                file_path_A=self.hparams.data_files['train'],
                file_path_B=self.hparams.mc_files['train'],
                **common_args
            )
            self.val_dataset = PairedFastDataset(
                file_path_A=self.hparams.data_files['val'],
                file_path_B=self.hparams.mc_files['val'],
                **common_args
            )
            print(f"Setup complete. Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")

            self._calculate_normalization_stats()

    def _calculate_normalization_stats(self):
        """Calculates and saves shared mean/std for constituents and conditions."""
        print(f"Calculating shared normalization stats from {self.hparams.num_batches_for_stats} training batches...")
        
        temp_loader = self._get_dataloader(self.train_dataset, shuffle=True)
        
        # Get feature counts from the first batch
        _, _, temp_real, _, _, _ = next(iter(temp_loader))
        num_const_features = temp_real.shape[-1]
        
        temp_loader = self._get_dataloader(self.train_dataset, shuffle=True)

        const_sum = torch.zeros(num_const_features)
        const_sq_sum = torch.zeros(num_const_features)
        const_count = torch.zeros(num_const_features)

        cond_sum = 0
        cond_sq_sum = 0
        total_jets = 0

        for i, batch in enumerate(tqdm(temp_loader, desc="Calculating Shared Stats")):
            if i >= self.hparams.num_batches_for_stats:
                break
            
            # MODIFIED: Unpack the new 6-item tuple which includes masks
            (jets_A, jets_B, real_A, real_B, mask_A, mask_B) = batch
            
            # MODIFIED: The normalization logic is now simpler and more robust.
            # `real_A` and `real_B` are already cleaned (NaNs are 0).
            # We use the provided masks to get the correct count of non-padded elements.
            # The mask is True for padded, so we invert it (~mask) to count real elements.
            
            # Sum the cleaned tensors. This is safe because NaNs are already 0.
            const_sum += real_A.sum(dim=(0, 1)) + real_B.sum(dim=(0, 1))
            const_sq_sum += (real_A ** 2).sum(dim=(0, 1)) + (real_B ** 2).sum(dim=(0, 1))
            
            # Count only the real, non-padded elements using the inverted mask.
            # We expand the mask to match the feature dimension for per-feature counting.
            valid_elements_A = (~mask_A).unsqueeze(-1).expand_as(real_A)
            valid_elements_B = (~mask_B).unsqueeze(-1).expand_as(real_B)
            const_count += valid_elements_A.sum(dim=(0, 1)) + valid_elements_B.sum(dim=(0, 1))
            
            # Condition (jet) stats are calculated as before
            cond_sum += jets_A.sum(dim=0) + jets_B.sum(dim=0)
            cond_sq_sum += (jets_A ** 2).sum(dim=0) + (jets_B ** 2).sum(dim=0)
            total_jets += jets_A.shape[0] * 2

        # Avoid division by zero if a feature has no valid elements
        const_count = torch.clamp(const_count, min=1)

        # Calculate and save the single, shared stats
        self.const_mean = const_sum / const_count
        const_var = (const_sq_sum / const_count) - (self.const_mean ** 2)
        self.const_std = torch.sqrt(torch.abs(const_var))

        self.cond_mean = cond_sum / total_jets
        cond_var = (cond_sq_sum / total_jets) - (self.cond_mean ** 2)
        self.cond_std = torch.sqrt(torch.abs(cond_var))

        self.const_std = torch.clamp(self.const_std, min=1e-8)
        self.cond_std = torch.clamp(self.cond_std, min=1e-8)

        print(f"Constituent Mean: {self.const_mean}")
        print(f"Constituent Std: {self.const_std}")
        print(f"Condition Mean: {self.cond_mean}")
        print(f"Condition Std: {self.cond_std}")
        print("Shared normalization stats calculated successfully.")
        
    def _get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            sampler=OriginalLikeBatchSampler(dataset, self.hparams.batch_size, shuffle=shuffle, drop_last=True),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)