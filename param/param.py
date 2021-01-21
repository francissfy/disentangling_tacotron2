import yaml


def read_config(yaml_file: str):
    f = open(yaml_file, "r")
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg
