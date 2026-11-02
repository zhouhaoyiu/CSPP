"""Read prospectively split, operationally packetized CWA development data."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class CWAOpenedPacketStoreSource:
    """Expose model-ready CWA event sequences without synthetic relocation."""

    preserve_event_context = True

    def __init__(
        self,
        name: str,
        hdf5_path: str | Path,
        split: str,
    ):
        import h5py

        self.name = str(name)
        self.hdf5_path = Path(hdf5_path)
        self.metadata_path = self.hdf5_path.with_suffix(".csv")
        if not self.hdf5_path.is_file():
            raise FileNotFoundError(self.hdf5_path)
        if not self.metadata_path.is_file():
            raise FileNotFoundError(self.metadata_path)
        all_metadata = pd.read_csv(self.metadata_path)
        required = {
            "store_index",
            "dataset_index",
            "event_id",
            "split",
            "stored_packet_count",
            "target_chunk",
            "source_magnitude",
            "source_distance_km",
        }
        missing = sorted(required - set(all_metadata.columns))
        if missing:
            raise ValueError(f"CWA packet metadata missing columns: {missing}")
        if all_metadata["store_index"].duplicated().any():
            raise ValueError("CWA packet store indices are duplicated")
        expected = np.arange(len(all_metadata), dtype=np.int64)
        if not np.array_equal(
            all_metadata["store_index"].to_numpy(dtype=np.int64), expected
        ):
            raise ValueError("CWA packet store indices are not contiguous")
        self.metadata = all_metadata
        split_indices = all_metadata.loc[
            all_metadata["split"].astype(str).eq(str(split)),
            "store_index",
        ].to_numpy(dtype=np.int64)
        if not len(split_indices):
            raise ValueError(f"CWA packet store has no {split!r} records")
        with h5py.File(self.hdf5_path, "r") as handle:
            quality = handle["quality_valid_mask"][:].astype(bool)
        stored_counts = all_metadata["stored_packet_count"].to_numpy(
            dtype=np.int64
        )
        uninterrupted = np.fromiter(
            (
                bool(quality[index, : stored_counts[index]].all())
                for index in split_indices
            ),
            dtype=bool,
            count=len(split_indices),
        )
        self.excluded_reset_indices = split_indices[~uninterrupted]
        self.event_indices = split_indices[uninterrupted]
        if not len(self.event_indices):
            raise ValueError(
                f"CWA packet store has no uninterrupted {split!r} records"
            )
        self.split = str(split)
        self._h5py = h5py
        self._handle = None

    @property
    def handle(self):
        if self._handle is None:
            self._handle = self._h5py.File(self.hdf5_path, "r")
        return self._handle

    def read_packetized(
        self,
        index: int,
        *,
        max_chunks: int,
        onset_response_packets: int,
        verifier_response_packets: int,
        verifier_start_delay_packets: int,
    ) -> dict:
        row = self.metadata.iloc[int(index)]
        if str(row["split"]) != self.split:
            raise ValueError("CWA packet source crossed its frozen split")
        stored_chunks = int(self.handle["chunks"].shape[1])
        if not 4 <= int(max_chunks) <= stored_chunks:
            raise ValueError(
                f"CWA packet store supports 4..{stored_chunks} chunks, "
                f"received {max_chunks}"
            )
        stored_target = int(row["target_chunk"])
        pre_context = min(140, int(max_chunks) - 4)
        crop_start = max(stored_target - pre_context, 0)
        crop_start = min(crop_start, stored_chunks - int(max_chunks))
        crop_stop = crop_start + int(max_chunks)
        chunks = self.handle["chunks"][
            int(index), crop_start:crop_stop
        ].astype(np.float32, copy=False)
        mask = self.handle["candidate_mask"][
            int(index), crop_start:crop_stop
        ].astype(np.float32, copy=False)
        target_chunk = stored_target - crop_start
        if not 0 <= target_chunk < int(max_chunks) - 3:
            raise ValueError("CWA target packet is outside the stored window")
        if mask[target_chunk] <= 0:
            raise ValueError("CWA target packet is not candidate-enabled")

        labels = np.zeros(max_chunks, dtype=np.float32)
        labels_event = np.zeros(max_chunks, dtype=np.float32)
        onset_end = min(
            target_chunk + int(onset_response_packets), max_chunks
        )
        verifier_start = min(
            target_chunk + int(verifier_start_delay_packets), max_chunks
        )
        verifier_end = min(
            target_chunk + int(verifier_response_packets), max_chunks
        )
        labels[target_chunk:onset_end] = mask[target_chunk:onset_end]
        labels_event[verifier_start:verifier_end] = mask[
            verifier_start:verifier_end
        ]
        target_offset = float(row.get("target_offset", math.nan))
        return {
            "chunks": torch.from_numpy(np.asarray(chunks)),
            "labels": torch.from_numpy(labels),
            "labels_event": torch.from_numpy(labels_event),
            "mask": torch.from_numpy(mask),
            "target_chunk": target_chunk,
            "target_offset": target_offset,
        }


__all__ = ["CWAOpenedPacketStoreSource"]
