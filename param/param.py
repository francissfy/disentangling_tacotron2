import yaml
import pprint


def load_config(yaml_file: str):
    f = open(yaml_file, "r")
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg


def print_cfg(cfg):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)


if __name__ == "__main__":
    yaml_file = "/Users/francis/code/disentangling_tacotron2/param/config.yaml"
    cfg = load_config(yaml_file)
    print_cfg(cfg)