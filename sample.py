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
        text_embedding = ema_mardm.encode_text(prompt_list)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)
        token_lens = Categorical(probs).sample()
    else:
        token_lens = torch.LongTensor(length_list) // 4
        token_lens = token_lens.to(device).long()
    m_length = token_lens * 4
    captions = prompt_list

    sample = 0
    kinematic_chain = kit_kinematic_chain if args.dataset_name == 'kit' else t2m_kinematic_chain

    for r in range(args.repeat_times):
        print("-->Repeat %d" % r)
        with torch.no_grad():
            pred_latents = ema_mardm.generate(captions, token_lens, args.time_steps, args.cfg,
                                              temperature=args.temperature, hard_pseudo_reorder=args.hard_pseudo_reorder)
            pred_motions = ae.decode(pred_latents)
            pred_motions = pred_motions.detach().cpu().numpy()
            data = pred_motions * std + mean

        for k, (caption, joint_data) in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d" % (k, caption, m_length[k]))
            s_path = pjoin(result_dir, str(k))
            os.makedirs(s_path, exist_ok=True)
            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), nb_joints).numpy()
            print(
                "joint stats:",
                "min", np.nanmin(joint),
                "max", np.nanmax(joint),
                "nan?", np.isnan(joint).any(),
                "inf?", np.isinf(joint).any()
            )
            save_path = pjoin(s_path, "caption:%s_sample%d_repeat%d_len%d.mp4" % (caption, k, r, m_length[k]))
            plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
            np.save(pjoin(s_path, "caption:%s_sample%d_repeat%d_len%d.npy" % (caption, k, r, m_length[k])), joint)


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