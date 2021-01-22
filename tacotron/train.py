import time
import argparse
import os
import math
import torch
from tacotron.tacotron import TCTRN_Tacotron
from tacotron.loss import TCTRN_TacotronLoss
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils.distributed import apply_gradient_allreduce
from utils.logger import TCTRN_Logger
from utils.dataloader import TextMelLoader, TextMelCollate
from numpy import finfo
from param.param import load_config


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def load_model(cfg):
    model = TCTRN_Tacotron(cfg)
    if cfg["EXPERIMENT"]["FP_16RUN"]:
        model.decoder.attention_layer.score_mask_value = finfo("float16").min
    if cfg["EXPERIMENT"]["DISTRIBUTED_RUN"]:
        model = apply_gradient_allreduce(model)

    return model


def init_distributed_run(cfg, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    torch.cuda.set_device(rank % torch.cuda.device_count())

    dist.init_process_group(
        backend=cfg["EXPERIMENT"]["DIST_BACKEND"],
        init_method=cfg["EXPERIMENT"]["DIST_URL"],
        world_size=n_gpus,
        rank=rank,
        group_name=group_name
    )

    print("Done initializing distributed")


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)
            os.chmod(output_directory, 0o775)
        logger = TCTRN_Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def prepare_dataloaders(cfg):
    trainset = TextMelLoader(cfg["DATA"]["TRAINING_FILES"], cfg)
    valset = TextMelLoader(cfg["DATA"]["VALIDATION_FILES"], cfg)
    collate_fn = TextMelCollate(cfg["TCTRN"]["DECODER"]["N_FRAMES_PER_STEP"])
    if cfg["EXPERIMENT"]["DISTRIBUTED_RUN"]:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        trainset, num_workers=1, shuffle=shuffle,
        sampler=train_sampler, batch_size=cfg["OPT"]["BATCH_SIZE"],
        pin_memory=False, drop_last=True, collate_fn=collate_fn
    )
    return train_loader, valset, collate_fn


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict["state_dict"]
    if len(ignore_layers) > 0:
        # TODO optimize
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def laod_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({
        "iteration": iteration,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate
    }, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus, collate_fn, logger,
             distributed_run, rank):
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(
            valset, sampler=val_sampler, num_workers=1, shuffle=False, batch_size=batch_size,
            pin_memory=False, collate_fn=collate_fn
        )

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i+1)
    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)


def train_tacotron(output_directory, log_directory, checkpoint_path, warm_start, n_gpus, rank, group_name, cfg):
    """Training and validation logging results to tensorboard and stdout
    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    cfg (object): comma separated list of "name=value" pairs.
    """
    # experimental parameters
    distributed_run = cfg["EXPERIMENT"]["DISTRIBUTED_RUN"]
    fp_16run = cfg["EXPERIMENT"]["FP_16RUN"]

    if distributed_run:
        init_distributed_run(cfg, n_gpus, rank, group_name)

    manual_seed = cfg["EXPERIMENT"]["SEED"]
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)

    model = load_model(cfg)
    learning_rate = cfg["OPT"]["LEARNING_RATE"]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=cfg["OPT"]["WEIGHT_DECAY"])

    if fp_16run:
        # TODO can remove fp16 run
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2"
        )

    if distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = TCTRN_TacotronLoss()

    logger = prepare_directories_and_logger(output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(cfg)

    # load ckpt
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(checkpoint_path, model, cfg["EXPERIMENT"]["IGNORE_LAYERS"])
        else:
            model, optimizer, _learning_rate, iteration = laod_checkpoint(checkpoint_path, model, optimizer)
            if cfg["EXPERIMENT"]["USE_SAVED_LEARNING_RATE"]:
                learning_rate = _learning_rate
            iteration += 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, cfg["EXPERIMENT"]["EPOCH"]):
        print(f"Epoch: {epoch}")
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            # TODO
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus)
            else:
                reduced_loss = loss.item()
            if fp_16run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if fp_16run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), cfg["OPT"]["GRAD_CLIP_THRESH"])
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["OPT"]["GRAD_CLIP_THRESH"])

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % cfg["EXPERIMENT"]["ITERS_PER_CHECKPOINT"] == 0):
                validate(model, criterion, valset, iteration,
                         cfg["OPT"]["BATCH_SIZE"], n_gpus, collate_fn, logger,
                         distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        required=True, help='configuration yamel file')
    config_file = parser.parse_args()
    cfg = load_config(config_file.config)

    train_param = cfg["TRAIN"]
    output_dir = train_param["OUTPUT_DIR"]
    log_dir = train_param["LOG_DIR"]
    checkpoint_path = train_param["CHECKPOINT_PATH"]
    warm_start = train_param["WARM_START"]
    n_gpus = train_param["N_GPUS"]
    rank = train_param["RANK"]
    group_name = train_param["GROUP_NAME"]

    train_tacotron(output_dir, log_dir, checkpoint_path, warm_start, n_gpus, rank, group_name, cfg)



