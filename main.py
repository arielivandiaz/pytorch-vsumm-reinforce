from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from utils import Logger, read_json, write_json, save_checkpoint, set_args
from models import *

from rewards import compute_reward
import vsum_tools


args = set_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

use_gpu = torch.cuda.is_available()

if args.use_cpu: use_gpu = False

def main():
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')
    num_videos = len(dataset.keys())
    splits = read_json(args.split)
    assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
    split = splits[args.split_id]
    train_keys = split['train_keys']
    test_keys = split['test_keys']
    print("# total videos {}. # train videos {}. # test videos {}".format(num_videos, len(train_keys), len(test_keys)))

    print("Initialize model")
    model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)

    #model = DNC(input_size=args.input_dim,hidden_size=args.hidden_dim)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
    else:
        start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        evaluate(model, dataset, test_keys, use_gpu)
        return

    print("==> Start training")
    start_time = time.time()
    model.train()
    baselines = {key: 0. for key in train_keys} # baseline rewards for videos
    reward_writers = {key: [] for key in train_keys} # record reward changes for each video

    for epoch in range(start_epoch, args.max_epoch):
        idxs = np.arange(len(train_keys))
        np.random.shuffle(idxs) # shuffle indices

        for idx in idxs:
            key = train_keys[idx]
            seq = dataset[key]['features'][...] # sequence of features, (seq_len, dim)
            seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)
            if use_gpu: seq = seq.cuda()
            probs = model(seq) # output shape (1, seq_len, 1)

            cost = args.beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term [Eq.11]
            m = Bernoulli(probs)
            epis_rewards = []
            for _ in range(args.num_episode):
                actions = m.sample()
                log_probs = m.log_prob(actions)
                reward = compute_reward(seq, actions, use_gpu=use_gpu)
                expected_reward = log_probs.mean() * (reward - baselines[key])
                cost -= expected_reward # minimize negative expected reward
                epis_rewards.append(reward.item())

            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
            reward_writers[key].append(np.mean(epis_rewards))

        epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
        print("epoch {}/{}\t reward {}\t".format(epoch+1, args.max_epoch, epoch_reward))

    write_json(reward_writers, osp.join(args.save_dir, 'rewards.json'))
    evaluate(model, dataset, test_keys, use_gpu)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
    model_save_path = osp.join(args.save_dir, 'model_epoch' + str(args.max_epoch) + '.pth.tar')
    save_checkpoint(model_state_dict, model_save_path)
    print("Model saved to {}".format(model_save_path))

    dataset.close()

def evaluate(model, dataset, test_keys, use_gpu):
    print("==> Test")
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'

        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]

            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            if args.verbose:
                table.append([key_idx+1, key, "{:.1%}".format(fm)])

            if args.save_results:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)

    if args.verbose:
        print(tabulate(table))

    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    print("Average F-score {:.1%}".format(mean_fm))

    return mean_fm

if __name__ == '__main__':
    main()
