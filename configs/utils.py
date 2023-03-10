import yaml

def load_config():
    with open("../configs/slim.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg