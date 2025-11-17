# ---------------------------
# SCRIPT: data.py
# ---------------------------

# Standard library
import os
import yaml
from pathlib import Path
from typing import Dict, List

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

# --- 1. The High-Performance Sampler ---
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

# --- 2. The High-Performance Constituent Dataset ---
class FastOriginalDataset(Dataset):
    """
    A PyTorch Dataset designed for fast, parallelized reading from a single HDF5 file.
    Each worker opens its own file handle, and it reads data in batches (slices).
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

    def __getitem__(self, slice_obj: slice):
        if self.file is None:
            self._initialize_worker()

        batch_size = slice_obj.stop - slice_obj.start
        inputs = {}

        for name, features in self.variables.items():
            batch_array = self.arrays[name]
            new_shape = (batch_size,) + self.dsets[name].shape[1:]
            batch_array.resize(new_shape, refcheck=False)
            self.dsets[name].read_direct(batch_array, source_sel=slice_obj)

            unstructured_array = s2u(batch_array[features], dtype=np.float32)
            inputs[name] = torch.from_numpy(unstructured_array)

        return inputs

# --- 3. The Paired Dataset for CycleGAN ---
class PairedFastDataset(Dataset):
    """
    A wrapper dataset that pairs two FastOriginalDataset instances for CycleGAN.
    It takes a slice and fetches the corresponding batch from both Domain A and Domain B.
    """
    def __init__(self, file_path_A: str, file_path_B: str, variables: dict, input_map: dict, constituent_name: str):
        self.dset_A = FastOriginalDataset(file_path_A, variables, input_map, constituent_name)
        self.dset_B = FastOriginalDataset(file_path_B, variables, input_map, constituent_name)
        self._len = min(len(self.dset_A), len(self.dset_B))

    def __len__(self):
        return self._len

    def __getitem__(self, slice_obj: slice):
        # Fetch a batch from each domain using the same slice
        batch_A = self.dset_A[slice_obj]
        batch_B = self.dset_B[slice_obj]

        jets_A = batch_A['jets']
        jets_B = batch_B['jets']
        constituents_A = batch_A['constituents']
        constituents_B = batch_B['constituents']
        
        # The model's training step only needs these four tensors
        return (jets_A, jets_B, constituents_A, constituents_B)

# --- 4. The Main High-Performance Lightning DataModule for CycleGAN ---
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

            # Calculate normalization stats using a subset of the training data
            self._calculate_normalization_stats()

    def _calculate_normalization_stats(self):
        """Calculates and saves shared mean/std for constituents and conditions."""
        print(f"Calculating shared normalization stats from {self.hparams.num_batches_for_stats} training batches...")
        
        temp_loader = self._get_dataloader(self.train_dataset, shuffle=True)

        const_sum = 0
        const_sq_sum = 0
        const_count = 0
        cond_sum = 0
        cond_sq_sum = 0
        total_jets = 0

        for i, batch in enumerate(tqdm(temp_loader, desc="Calculating Shared Stats")):
            if i >= self.hparams.num_batches_for_stats:
                break
            
            (jets_A, jets_B, real_A, real_B) = batch
            
            mask_A_expanded = ~(real_A[:, :, 0] == 0).unsqueeze(-1)
            mask_B_expanded = ~(real_B[:, :, 0] == 0).unsqueeze(-1)
            
            # Accumulate from both A and B domains into shared variables
            valid_A = real_A * mask_A_expanded
            valid_B = real_B * mask_B_expanded
            
            const_sum += valid_A.sum(dim=(0, 1)) + valid_B.sum(dim=(0, 1))
            const_sq_sum += (valid_A ** 2).sum(dim=(0, 1)) + (valid_B ** 2).sum(dim=(0, 1))
            const_count += mask_A_expanded.sum(dim=(0, 1)) + mask_B_expanded.sum(dim=(0, 1))
            
            cond_sum += jets_A.sum(dim=0) + jets_B.sum(dim=0)
            cond_sq_sum += (jets_A ** 2).sum(dim=0) + (jets_B ** 2).sum(dim=0)
            total_jets += jets_A.shape[0] * 2

        # Calculate and save the single, shared stats
        self.const_mean = const_sum / const_count
        const_var = (const_sq_sum / const_count) - (self.const_mean ** 2)
        self.const_std = torch.sqrt(torch.abs(const_var)) # Use abs for stability

        self.cond_mean = cond_sum / total_jets
        cond_var = (cond_sq_sum / total_jets) - (self.cond_mean ** 2)
        self.cond_std = torch.sqrt(torch.abs(cond_var))

        print("Shared normalization stats calculated successfully.")
        
    def _get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=None,  # Crucial for the custom sampler
            sampler=OriginalLikeBatchSampler(dataset, self.hparams.batch_size, shuffle=shuffle, drop_last=True),
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)