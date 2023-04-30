import pickle
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

def create_metadata_sequential(metadata_path):
    metadata_path = Path(metadata_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    metadata_seq = {"look_up_table": [],
                    "flows_information": metadata["flows_information"]}

    foursome = [0 for _ in range(4)]

    pair_before = None
    pair_before_before = None
    for pair in tqdm(metadata["look_up_table"]):
        if pair_before is not None and pair_before_before is not None:

            foursome[0] = pair[0]
            foursome[1] = pair[1]
            foursome[2] = pair_before[1]
            foursome[3] = pair_before_before[1]

            sequence_check = True
            name_first = foursome[0][0].split("_")[:-1]
            number_first = int(foursome[0][0].split("_")[-1].split(".")[0])
            for pcl in foursome:
                name = pcl[0]
                number = int(name.split("_")[-1].split(".")[0])
                name = name.split("_")[:-1]

                if number_first != number or name != name_first:
                    sequence_check = False
                    break

                number_first -= 1

            if sequence_check == True:
                metadata_seq["look_up_table"].append(deepcopy(foursome))

        #if pair_before is not None:
        pair_before_before = pair_before
        pair_before = pair

    with open(metadata_path.parent / "metadata_seq", 'wb') as file:
        pickle.dump(metadata_seq, file)

if __name__ == "__main__":
    create_metadata_sequential("../../data/waymoflow/test/metadata")