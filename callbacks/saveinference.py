import numpy as np
from pytorch_lightning.callbacks import Callback
import os
import torch
from pathlib import Path

class SaveInference(Callback):
    def __init__(self, dirpath: str, every_n_train_steps: int = 100, every_n_test_steps: int = 1):
        super(SaveInference, self).__init__()
        self.every_n_train_steps = every_n_train_steps
        self.every_n_test_steps = every_n_test_steps
        self.show = False

        self.path = Path(dirpath)
        os.makedirs(self.path, exist_ok=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        if batch_idx % self.every_n_test_steps == 0:

            aggregated_flow = pl_module.last_output[3]["aggregated_flow"].detach().cpu().numpy()
            np.save(self.path / f"{str(batch_idx).zfill(6)}", aggregated_flow)



