import os
from os.path import join as pjoin
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import random
from models.AE import AE_models
from models.MARDM import MARDM_models
from models.LengthEstimator import LengthEstimator
from utils.motion_process import recover_from_ric, plot_3d_motion, kit_kinematic_chain, t2m_kinematic_chain
import argparse

import glob
from pathlib import Path


import re

def safe_stem(s: str, max_len: int = 120) -> str:
    # Replace path separators explicitly (Linux/macOS: '/', Windows: '\')
    s = s.replace("/", "_").replace("\\", "_")

    # Replace anything not filename-friendly with underscores
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    # Collapse runs of underscores and trim
    s = re.sub(r"_+", "_", s).strip("._-")

    # Avoid overly long filenames (common filesystem limit is 255 bytes)
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-")

    return s or "caption"


def chunked(xs: list[int], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def main(args):
    #################################################################################
    #                                      Seed                                     #
    #################################################################################
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # setting this to true significantly increase training and sampling speed
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    #################################################################################
    #                                       Data                                    #
    #################################################################################
    dim_pose = 64 if args.dataset_name == 'kit' else 67
    nb_joints = 21 if args.dataset_name == 'kit' else 22
    data_root = f'{args.dataset_dir}/KIT-ML/' if args.dataset_name == 'kit' else f'{args.dataset_dir}/HumanML3D/'
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    #################################################################################
    #                                      Models                                   #
    #################################################################################
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    result_dir = pjoin('./generation', args.name)
    os.makedirs(result_dir, exist_ok=True)

    ae = AE_models[args.ae_model](input_width=dim_pose)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, args.ae_name, 'model',
                            'latest.tar' if args.dataset_name == 't2m' else 'net_best_fid.tar'), map_location='cpu')
    model_key = 'ae'
    ae.load_state_dict(ckpt[model_key])

    ema_mardm = MARDM_models[args.model](ae_dim=ae.output_emb_width, cond_mode='text')
    model_dir = pjoin(model_dir, 'latest.tar')
    checkpoint = torch.load(model_dir, map_location='cpu')
    missing_keys2, unexpected_keys2 = ema_mardm.load_state_dict(checkpoint['ema_mardm'], strict=False)
    assert len(unexpected_keys2) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys2])

    length_estimator = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(args.checkpoints_dir, args.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location='cpu')
    length_estimator.load_state_dict(ckpt['estimator'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################################################################
    #                                     Sampling                                  #
    #################################################################################
    prompt_list = []
    length_list = []

    est_length = False
    if args.text_prompt != "":
        prompt_list.append(args.text_prompt)
        if args.motion_length == 0:
            est_length = True
        else:
            length_list.append(args.motion_length)
    elif args.text_path != "":
        with open(args.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                infos = line.split('#')
                prompt_list.append(infos[0])
                if len(infos) == 1 or (not infos[1].isdigit()):
                    est_length = True
                    length_list = []
                else:
                    length_list.append(int(infos[-1]))
    else:
        raise "A text prompt, or a file a text prompts are required!!!"

    ae.to(device)
    ema_mardm.to(device)
    length_estimator.to(device)

    ae.eval()
    ema_mardm.eval()
    length_estimator.eval()

    if est_length:
        print("Since no motion length are specified, we will use estimated motion lengthes!!")
        with torch.no_grad():
            text_embedding = ema_mardm.encode_text(prompt_list)
            pred_dis = length_estimator(text_embedding)
            probs = F.softmax(pred_dis, dim=-1)
            token_lens_all = Categorical(probs).sample()
    else:
        token_lens_all = torch.LongTensor(length_list) // 4
    
    token_lens_all = token_lens_all.to(device).long()
    m_length_all = token_lens_all * 4
    captions = prompt_list

    sample = 0
    kinematic_chain = kit_kinematic_chain if args.dataset_name == 'kit' else t2m_kinematic_chain

    def is_done(idx: int) -> bool:
        s_path = Path(result_dir) / str(idx)
        if not s_path.exists():
            return False
        return (len(list(s_path.glob("*.mp4"))) > 0) or (len(list(s_path.glob("*.npy"))) > 0)

    pending_indices = [i for i in range(len(prompt_list)) if not is_done(i)]
    if not pending_indices:
        print(f"All {len(prompt_list)} prompts already have outputs under {result_dir}. Nothing to do.")
        return
    
    captions = [prompt_list[i] for i in pending_indices]
    token_lens = token_lens_all[pending_indices]
    m_length = m_length_all[pending_indices]

    print(f"Found {len(prompt_list) - len(pending_indices)} completed; generating {len(pending_indices)} prompts:")
    print(f"First few pending indices: {pending_indices[:10]}")

    batch_size = 8
    batch_id_global = 0

    for r in range(args.repeat_times):
        print("-->Repeat %d" % r)

        for batch_indices in chunked(pending_indices, batch_size):
            batch_id_global += 1
            print(f"  -> Batch {batch_id_global} (size={len(batch_indices)}): {batch_indices[:10]}")

            batch_captions = [prompt_list[i] for i in batch_indices]
            batch_token_lens = token_lens_all[batch_indices]
            batch_m_length = m_length_all[batch_indices]

            with torch.no_grad():
                pred_latents = ema_mardm.generate(
                    batch_captions,
                    batch_token_lens,
                    args.time_steps,
                    args.cfg,
                    temperature = args.temperature,
                    hard_pseudo_reorder = args.hard_pseudo_reorder
                )
                pred_motions = ae.decode(pred_latents)
                pred_motions = pred_motions.detach().cpu().numpy()
                data = pred_motions * std + mean

            for local_i, (orig_idx, caption, joint_data) in enumerate(zip(batch_indices, batch_captions, data)):
                ml = int(batch_m_length[local_i])
                print(f"    ----> Prompt index {orig_idx}: {caption}  len={ml}")
                s_path = pjoin(result_dir, str(orig_idx))
                os.makedirs(s_path, exist_ok=True)
                joint_data = joint_data[:ml]
                joint = recover_from_ric(torch.from_numpy(joint_data).float(), nb_joints).numpy()
                print(
                    "    joint stats:",
                    "min", np.nanmin(joint),
                    "max", np.nanmax(joint),
                    "nan?", np.isnan(joint).any(),
                    "inf?", np.isinf(joint).any()
                )
                cap_stem = safe_stem(caption)
                mp4_name = f"{cap_stem}_sample{orig_idx}_repeat{r}_len{ml}.mp4"
                npy_name = f"{cap_stem}_sample{orig_idx}_repeat{r}_len{ml}.npy"
                save_mp4 = pjoin(s_path, mp4_name)
                save_npy = pjoin(s_path, npy_name)
                plot_3d_motion(save_mp4, kinematic_chain, joint, title=caption, fps=20)
                np.save(save_npy, joint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='MARDM')
    parser.add_argument('--ae_name', type=str, default="AE")
    parser.add_argument('--ae_model', type=str, default='AE_Model')
    parser.add_argument('--model', type=str, default='MARDM-SiT-XL')
    parser.add_argument('--dataset_name', type=str, default='t2m')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument("--time_steps", default=18, type=int)
    parser.add_argument("--cfg", default=4.5, type=float)
    parser.add_argument("--temperature", default=1, type=float)

    parser.add_argument('--text_prompt', default='', type=str)
    parser.add_argument('--text_path', type=str, default="")
    parser.add_argument("--motion_length", default=0, type=int)
    parser.add_argument("--repeat_times", default=1, type=int)
    parser.add_argument('--hard_pseudo_reorder', action="store_true")
    arg = parser.parse_args()
    main(arg)