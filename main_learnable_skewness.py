import os
import sys
import logging
import argparse
import random
import math
import json
import time
import itertools
import copy
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import redirect_stdout
from config import Config
from models.model_LRA import ModelForSC, ModelForSCDual
from models.dataset_LRA import LRADataset
import pickle
import requests
import json


def frobenius_distance(A, B):
    error = A - B
    squared_error = torch.pow(error, 2)
    sum_squared_error = torch.sum(squared_error)
    distance = torch.sqrt(sum_squared_error)

    return distance


def skewness(X):
    mu = np.mean(X)
    std = np.std(X)
    z = (X - mu) / std

    return np.mean(z ** 3)


def flood_fill(data, x, y, ff_mat, thresh):
    if x + 1 == data.shape[0] or y + 1 == data.shape[1]:
        return data, x, y, ff_mat, thresh
    else:
        if data[x + 1][y] > data[x][y + 1]:
            if data[x + 1][y] > data[x + 1][y + 1]:
                if data[x + 1][y] > thresh:
                    # print(f"x: {x}, y: {y}")
                    ff_mat[x + 1][y] = data[x + 1][y]
                data, x, y, ff_mat, thresh = flood_fill(data, x + 1, y, ff_mat, thresh)
            else:
                if data[x + 1][y + 1] > thresh:
                    # print(f"x: {x}, y: {y}")
                    ff_mat[x + 1][y + 1] = data[x + 1][y + 1]
                data, x, y, ff_mat, thresh = flood_fill(data, x + 1, y + 1, ff_mat, thresh)
        else:
            if data[x][y + 1] > data[x + 1][y + 1]:
                if data[x][y + 1] > thresh:
                    # print(f"x: {x}, y: {y}")
                    ff_mat[x][y + 1] = data[x][y + 1]
                data, x, y, ff_mat, thresh = flood_fill(data, x, y + 1, ff_mat, thresh)
            else:
                if data[x + 1][y + 1] > thresh:
                    # print(f"x: {x}, y: {y}")
                    ff_mat[x + 1][y + 1] = data[x + 1][y + 1]
                data, x, y, ff_mat, thresh = flood_fill(data, x + 1, y + 1, ff_mat, thresh)
    return data, x, y, ff_mat, thresh


logger = logging.getLogger(__name__)
device_ids = list(range(torch.cuda.device_count()))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validation(model, mat_lst, ds_iter, total_step, training_config, model_config, checkpoint_path, global_step,
               best_dev_accu, writer, task, init_t, update):
    val_acc = []
    eval_losses = AverageMeter()
    eval_attn_losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for dev_step_idx in range(training_config["num_eval_steps"]):
            _, batch = next(ds_iter['dev'])
            if task == 'lra-retrieval':
                input_ids_0 = batch['input_ids_0'].cuda()
                mask_0 = batch['mask_0'].cuda()
                input_ids_1 = batch['input_ids_1'].cuda()
                mask_1 = batch['mask_1'].cuda()
                # print(mask[0])
                label = batch['label'].cuda()
                outputs, attn_lst = model(input_ids_0, input_ids_1, mask_0, mask_1, label, mat_lst, False)
                if len(mat_lst) == 0:
                    attn_loss_0 = sum(sum(outputs["attn_loss_0"])) / 4 * model_config["num_layers"] * training_config[
                        "attn_loss_scale"]
                    attn_loss_1 = sum(sum(outputs["attn_loss_1"])) / 4 * model_config["num_layers"] * training_config[
                        "attn_loss_scale"]
                    attn_loss = (attn_loss_0 + attn_loss_1) / 2
                else:
                    attn_loss = 0
            else:
                input = batch['input_ids_0'].cuda()
                mask = batch['mask_0'].cuda()
                label = batch['label'].cuda()
                outputs, attn_lst = model(input, mask, label, mat_lst, False)
                if len(mat_lst) == 0:
                    attn_loss = sum(sum(outputs["attn_loss"])) / 4 * model_config["num_layers"] * training_config[
                        "attn_loss_scale"]
                    # attn_loss = sum(outputs["attn_loss"])*model_config["num_layers"] * training_config["attn_loss_scale"]
                else:
                    attn_loss = 0
            loss = outputs["loss"].mean()
            eval_losses.update(loss.mean())
            eval_attn_losses.update(attn_loss)
            acc = outputs["accu"].mean()
            val_acc.append(acc)

        total_acc = sum(val_acc) / len(val_acc)
        if total_acc > best_dev_accu:
            # best_dev_accu = total_acc
            if (global_step + 1) >= 100:
                best_dev_accu = total_acc
                update += 1
                torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
                print('best model saved: step = ', global_step, 'dev accu = ', total_acc)

        writer.add_scalar('val/loss', eval_losses.avg, global_step)
        writer.add_scalar('val/accu', total_acc, global_step)
        writer.add_scalar('val/attn_loss', eval_attn_losses.avg, global_step)

    print("\nValidation Results")
    print("Global Steps: %d" % global_step)
    print("Valid Loss: %2.5f" % eval_losses.avg)
    print("Valid Accuracy: %2.5f" % total_acc)
    print("time stamp: {}".format((time.time() - init_t)))

    return best_dev_accu, update


def train_step(model, optimizer, lr_scheduler, ds_iter, amp_scaler, training_config, model_config, writer, task,
               global_name):
    logger.info("***** Running training *****")
    logger.info("  Total steps = %d", training_config["num_train_steps"])
    losses = AverageMeter()

    checkpoint_path = training_config['checkpoint_path']
    best_dev_accu = 0
    transition = False

    total_step = training_config["num_train_steps"]
    # training_config["num_dense_train_steps"] = 5000
    epoch_iterator = tqdm(ds_iter['train'],
                          desc="Training (X / X Steps) (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)

    model.train()
    init_t = time.time()

    mat_lst = [] # 稀疏注意力模式
    prev_l = [] # 前一注意力分布
    cur_l = [] # 当前注意力分布
    prev_dist = [] # 注意力变化幅度
    cur_dist = []
    mem_stat = [] # GPU 内存统计

    update = 0
    total_time = 0
    transition_step = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for step, batch in enumerate(epoch_iterator):
        if (step + 1) % training_config["eval_frequency"] == 0:
            is_attn = True
        else:
            is_attn = False

        if task == 'lra-retrieval':
            input_ids_0 = batch['input_ids_0'].cuda()
            mask_0 = batch['mask_0'].cuda()
            input_ids_1 = batch['input_ids_1'].cuda()
            mask_1 = batch['mask_1'].cuda()
            # print(mask[0])
            label = batch['label'].cuda()
            outputs, attn_lst = model(input_ids_0, input_ids_1, mask_0, mask_1, label, mat_lst, True)
            if len(mat_lst) == 0:
                attn_loss_0 = sum(sum(outputs["attn_loss_0"])) / 4 * model_config["num_layers"] * training_config[
                    "attn_loss_scale"]
                attn_loss_1 = sum(sum(outputs["attn_loss_1"])) / 4 * model_config["num_layers"] * training_config[
                    "attn_loss_scale"]
                attn_loss = (attn_loss_0 + attn_loss_1) / 2
            else:
                attn_loss = 0
        else:
            input = batch['input_ids_0'].cuda()
            mask = batch['mask_0'].cuda()
            # print(mask[0])
            label = batch['label'].cuda()
            # print(label)
            outputs, attn_lst = model(input, mask, label, mat_lst, True)  # 开始进入sddmm运算
            # print(outputs)
            if len(mat_lst) == 0:
                # attn_loss = sum(sum(outputs["attn_loss"]))/4*model_config["num_layers"] * training_config["attn_loss_scale"]
                attn_loss = sum(outputs["attn_loss"]) * model_config["num_layers"] * training_config["attn_loss_scale"]
            else:
                attn_loss = 0

        if len(mat_lst) == 0:
            dense_mem = torch.cuda.memory_stats()['active_bytes.all.allocated']
            dense_step = step
            dense_time = time.time() - init_t
            dense_attn = attn_lst

        else:
            sparse_mem = torch.cuda.memory_stats()['active_bytes.all.allocated']
            sparse_step = step
            sparse_attn = attn_lst

        mem_stat.append([step, torch.cuda.memory_stats()['active_bytes.all.allocated'] >> 20])

        loss = outputs["loss"].mean()
        # print(attn_loss)
        acc = outputs["accu"].mean()
        # attn_loss = outputs["attn_loss"] * training_config["attn_loss_scale"]
        # print(loss)

        amp_scaler.scale(loss + attn_loss).backward()  # loss.backward()
        amp_scaler.unscale_(optimizer)
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1)  # Gradient Clipping
        amp_scaler.step(optimizer)
        amp_scaler.update()
        lr_scheduler.step()
        losses.update(loss)
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (step, total_step, losses.val))

        if len(mat_lst) == 0:
            dense_step = step
            transition = True

            for l in range(model_config["num_layers"]):
                X = np.array(attn_lst[l].detach().cpu()).ravel()

                if len(prev_l) < model_config["num_layers"]:
                    prev_l.append(X)
                    cur_l.append(X)
                    prev_dist.append(np.sqrt(np.sum(np.power(prev_l[l] - cur_l[l], 2))))
                    cur_dist = copy.deepcopy(prev_dist)
                    transition = False

                else:
                    cur_l[l] = X
                    cur_dist[l] = np.sqrt(np.sum(np.power(prev_l[l] - cur_l[l], 2)))

                    if abs(cur_dist[l] - prev_dist[l]) / prev_dist[l] * 100 >= training_config['distance'] or prev_dist[
                        l] == 0:
                        transition = False

                skew = skewness(X)

                if skew < training_config['skewness']:
                    transition = False

            prev_dist = copy.deepcopy(cur_dist)
            prev_l = copy.deepcopy(cur_l)

            # if step == 1000:
            #     transition = True

        if (step + 1) % training_config["eval_frequency"] == 0:
            print(outputs["attn_loss"])

            if task == 'lra-retrieval' and (step + 1) % 1000 == 0:
                end.record()
                torch.cuda.synchronize()
                total_time += (start.elapsed_time(end))
                best_dev_accu, update = validation(model, mat_lst, ds_iter, total_step, training_config, model_config,
                                                   checkpoint_path, step, best_dev_accu, writer, task, init_t, update)
                # print(total_time)
                model.train()
                start.record()
            else:
                end.record()
                torch.cuda.synchronize()
                total_time += (start.elapsed_time(end))
                best_dev_accu, update = validation(model, mat_lst, ds_iter, total_step, training_config, model_config,
                                                   checkpoint_path, step, best_dev_accu, writer, task, init_t, update)
                # print(total_time)
                model.train()
                start.record()

        pattern_t = time.time()

        # if (step + 1) == training_config["num_dense_train_steps"]:
        if transition and len(mat_lst) == 0:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training_config["learning_rate"] * 0.5,
                betas=(0.9, 0.999), eps=1e-6, weight_decay=training_config["weight_decay"]
            )

            for name, p in model.named_parameters():
                if 'pattern' in name:
                    path = f'./pickle/{task}'
                    if not os.path.exists(path):
                        os.mkdir(path)
                    with open(f'{path}/{name}.pickle', 'wb') as f:
                        pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)
                        print(f"{name} saved")
                    p.requires_grad = False

            block_size = model_config["block_size"]
            num_blocks = int(model_config["max_seq_len"] / block_size)

            for l in range(model_config["num_layers"]):
                attn_pattern_path = f'./pickle/{task}/module.model.transformer_{l}.mha.pattern.pickle'

                print(attn_pattern_path)
                with open(attn_pattern_path, 'rb') as f:
                    attn_pattern = pickle.load(f)

                attn_pattern = torch.matmul(attn_pattern, attn_pattern.t())
                # attn_pattern = attn_pattern.unsqueeze(dim=0)
                # attn_pattern = avg_pool(attn_pattern)
                attn_pattern = attn_pattern.view(-1)
                topk = torch.topk(attn_pattern, int(attn_pattern.shape[0] * model_config["density"]))[1]
                print(topk)
                mat = torch.zeros((num_blocks * num_blocks), dtype=torch.float32, device='cuda')
                mat[topk] = 1
                mat = mat.reshape(num_blocks, num_blocks)
                print(torch.count_nonzero(mat))

                mat_lst.append(mat)
                module_name = f"transformer_{l}"
                transformer_module = getattr(model.module.model, module_name)
                mha_module = getattr(transformer_module, "mha")
                mha_module.reconstruct_for_blocksparse(model_config)

            mat_lst = torch.stack(mat_lst, 0)
            # mat_lst = torch.stack([mat_lst,mat_lst,mat_lst,mat_lst], 0)
            mat_lst = torch.stack([mat_lst], 0)

            # print(layer_attn.device)
            # print(layer_attn.shape)

            print("total pattern searching time (s): {}".format(time.time() - pattern_t))
            pickle_path = f'./pickle/layer_attn-{model_config["random_seed"]}'
            print(f"Model is located at {pickle_path}")

            if not os.path.exists(pickle_path):
                os.mkdir(pickle_path)

            print(f"Model is saved at {pickle_path}/mat_list_{task}_{global_name}.pickle")
            with open(f'{pickle_path}/mat_lst_{task}_{global_name}.pickle', 'wb') as f:
                pickle.dump(mat_lst, f, pickle.HIGHEST_PROTOCOL)

        #        if transition and step - dense_step == 10:
        #            break

        if (step + 2) > total_step:
            break

    print('total training step (k): {}'.format(total_step / 1000.0))
    print("total training time (s): {}".format(time.time() - init_t))
    print("total training time (ms): {}".format(total_time))
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak'] >> 20))
    print("allocated memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated'] >> 20))
    print(torch.cuda.memory_summary(device=device_ids))
    print("dense memory(MB) : {}", dense_mem >> 20)
    print("dense step : {}", dense_step)
    print("dense time : {}", dense_time)
    print("sparse memory(MB) : {}", sparse_mem >> 20)
    print("sparse step : {}", sparse_step - dense_step)
    print("sparse_time : {}", time.time() - dense_time)

    return model, mat_lst


def evaluation(model, mat_lst, ds_iter, training_config, task):
    val_acc = []
    eval_losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        for _, batch in ds_iter['test']:
            if task == 'lra-retrieval':
                input_ids_0 = batch['input_ids_0'].cuda()
                mask_0 = batch['mask_0'].cuda()
                input_ids_1 = batch['input_ids_1'].cuda()
                mask_1 = batch['mask_1'].cuda()
                # print(mask[0])
                label = batch['label'].cuda()
                outputs, attn_lst = model(input_ids_0, input_ids_1, mask_0, mask_1, label, mat_lst, False)
            else:
                input = batch['input_ids_0'].cuda()
                mask = batch['mask_0'].cuda()
                # print(mask[0])
                label = batch['label'].cuda()
                outputs, attn_lst = model(input, mask, label, mat_lst, False)
            loss = outputs["loss"].mean()
            eval_losses.update(loss.mean())
            acc = outputs["accu"].mean()
            val_acc.append(acc)

        total_acc = sum(val_acc) / len(val_acc)

    print("Evaluation Results")
    print("Loss: %2.5f" % eval_losses.avg)
    print("Accuracy: %2.5f" % total_acc)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        help="train eval")
    parser.add_argument("--checkpoint", type=str, default="test",
                        help="load ./checkpoints/model_name.model to evaluation")
    parser.add_argument("--task", type=str, default="lra-image",
                        help="lra-listops, lra-retrieval, lra-text, lra-pathfinder32-curv_contour_length_14")
    parser.add_argument('--random', type=int, default=0)
    parser.add_argument('--name', type=str)
    parser.add_argument('--sk', type=float, default=1.7)
    parser.add_argument('--ds', type=float, default=1.3)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    if args.task == 'lra-pathfinder':
        args.task = 'lra-pathfinder32-curv_contour_length_14'

    ### get model config ###
    model_config = Config[args.task]["model"]
    model_config["mixed_precision"] = True
    model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))
    model_config["random_seed"] = args.random
    model_config["task"] = args.task
    training_config = Config[args.task]["training"]
    training_config['skewness'] = args.sk
    training_config['distance'] = args.ds

    ### log preparation ###
    log_dir = './logs/log-{}/'.format(args.random)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, args.task)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_path = os.path.join(log_dir, '{}.{}.log'.format(args.mode, args.name))
    redirect_stdout(open(log_path, 'w'))

    writer = SummaryWriter(os.path.join(log_dir, '{}.tensorboard'.format(args.name)))

    ###  set the random seeds for deterministic results. ####
    SEED = args.random
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_ids = list(range(torch.cuda.device_count()))
    model_config['batch_size'] = int(training_config['batch_size'] / len(device_ids))

    print(f"GPU list: {device_ids}")
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    print(json.dumps([model_config, training_config], indent=4))

    ### model preparation ###
    if args.task == "lra-retrieval":
        model = ModelForSCDual(model_config)
    else:
        model = ModelForSC(model_config)

    model = nn.DataParallel(model, device_ids=device_ids)

    checkpoint_dir = './checkpoints/checkpoints-{}/'.format(args.random)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, args.task)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, '{}.model'.format(args.name))
    training_config["checkpoint_path"] = checkpoint_path

    model = model.cuda()
    print(model)
    print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush=True)
    print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush=True)

    ### data preparation ###

    ds_iter = {
        "train": DataLoader(LRADataset(f"data/lra_processed/{args.task}.train.pickle", True),
                            batch_size=training_config["batch_size"], drop_last=True),
        "dev": enumerate(DataLoader(LRADataset(f"data/lra_processed/{args.task}.dev.pickle", True),
                                    batch_size=training_config["batch_size"], drop_last=True)),
        "test": enumerate(DataLoader(LRADataset(f"data/lra_processed/{args.task}.test.pickle", False),
                                     batch_size=training_config["batch_size"], drop_last=True)),
    }

    ### training preparation ###

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        betas=(0.9, 0.999), eps=1e-6, weight_decay=training_config["weight_decay"]
    )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=training_config["learning_rate"],
        pct_start=training_config["warmup"] / training_config["num_train_steps"],
        anneal_strategy=training_config["lr_decay"],
        total_steps=training_config["num_train_steps"]
    )

    amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None

    accumu_steps = model_config["bz_rate"] if "bz_rate" in model_config else 1
    print(f"accumu_steps={accumu_steps}")
    training_config['accumu_steps'] = accumu_steps

    ### train ###
    if args.mode == 'train':
        model, mat_lst = train_step(model, optimizer, lr_scheduler, ds_iter, amp_scaler,
                                    training_config, model_config, writer, args.task, args.name)

    ### eval ###
    if os.path.exists(checkpoint_path) and checkpoint_path != './checkpoints/test.model':
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading the best model from: " + checkpoint_path)

    evaluation(model, mat_lst, ds_iter, training_config, args.task)


if __name__ == '__main__':
    main()
