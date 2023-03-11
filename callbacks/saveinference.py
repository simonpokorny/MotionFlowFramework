import numpy as np
from pytorch_lightning.callbacks import Callback
import os

class SaveInference(Callback):
    def __init__(self, dirpath: str, name:str="lightning_logs", version:int=None, every_n_train_steps: int = 100):
        super(SaveInference, self).__init__()
        self.every_n_train_steps = every_n_train_steps

        self.path = os.path.join(dirpath, name, f'version_{version}', 'train_output')
        os.makedirs(self.path, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.every_n_train_steps:
            outputs = outputs.detach().cpu().numpy()
            np.savez(os.path.join(self.path, f'{str(batch_idx).zfill(8)}.npz'), outputs)