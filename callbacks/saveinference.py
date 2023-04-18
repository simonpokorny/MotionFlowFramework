import numpy as np
from pytorch_lightning.callbacks import Callback
import os
import torch

class SaveInference(Callback):
    def __init__(self, dirpath: str, name:str="lightning_logs", version:int=None, every_n_train_steps: int = 100):
        super(SaveInference, self).__init__()
        self.every_n_train_steps = every_n_train_steps

        self.path = os.path.join(name, f'version_{version}', 'train_output')
        os.makedirs(self.path, exist_ok=False)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx % self.every_n_train_steps == 0:

            (previous_batch, current_batch), _, trans_matrix = batch
            previous_batch_pc, previous_batch_grid, previous_batch_mask = previous_batch
            current_batch_pc, current_batch_grid, current_batch_mask = current_batch

            P_T_C = trans_matrix
            C_T_P = torch.linalg.inv(P_T_C)

            curr_path = os.path.join(self.path, f'{str(batch_idx).zfill(8)}')
            os.makedirs(curr_path, exist_ok=True)

            np.save(os.path.join(curr_path, f'previous_pcl.npy'), previous_batch_pc.detach().cpu().numpy())
            np.save(os.path.join(curr_path, f'previous_mask.npy'), previous_batch_mask.detach().cpu().numpy())
            np.save(os.path.join(curr_path, f'previous_grid.npy'), previous_batch_grid.detach().cpu().numpy())

            np.save(os.path.join(curr_path, f'current_pcl.npy'), current_batch_pc.detach().cpu().numpy())
            np.save(os.path.join(curr_path, f'current_mask.npy'), current_batch_mask.detach().cpu().numpy())
            np.save(os.path.join(curr_path, f'current_grid.npy'), current_batch_grid.detach().cpu().numpy())

            np.save(os.path.join(curr_path, f'P_T_C.npy'), P_T_C.detach().cpu().numpy())
            np.save(os.path.join(curr_path, f'C_T_P.npy'), C_T_P.detach().cpu().numpy())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        print(batch_idx, batch_idx % self.every_n_train_steps)
        if batch_idx % self.every_n_train_steps == 0:
            outputs = outputs.detach().cpu().numpy()
            np.savez(os.path.join(self.path, f'{str(batch_idx).zfill(8)}.npz'), outputs)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        pass
    def on_predict_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        pass