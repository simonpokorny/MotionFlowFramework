from tqdm import tqdm

from configs import get_config
from datasets import get_datamodule
from datasets.visualization.visualizer import Visualizer

if __name__ == "__main__":

    DATASET = "nuscenes"
    ITER_OVER = "train"

    cfg = get_config("../configs/slim.yaml", dataset=DATASET)
    data_module = get_datamodule(dataset=DATASET, data_path=None, cfg=cfg)
    data_module.setup()

    if ITER_OVER == "train":
        dl = data_module.train_dataloader()
    elif ITER_OVER == "test":
        dl = data_module.test_dataloader()
    else:
        raise ValueError()

    dl.num_workers = 0
    # Wrap the dataloader into visualizer
    dl = Visualizer(dl, visualize="flow3d")

    for idx, (x, flow, T_gt) in enumerate(tqdm(dl)):
        continue