"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import cv2

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    print(dist_util.dev())
    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop_progressive
        )

        img = th.randn((args.batch_size, 3, args.image_size, args.image_size), device=dist_util.dev())

        one_step = []
        final = None
        stp = 0
        for spl in sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            noise=img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs
        ):
            stp += 1
            final = spl
            sample = final["sample"]

            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            sample = sample.cpu().numpy()
            tmp = []
            for _ in range(sample.shape[0]):
                tmp.append(cv2.resize(sample[_], (25, args.image_size), interpolation=cv2.INTER_CUBIC))
            one_step.append(tmp)
            logger.log(f"gen {stp} samples")
        one_step_arr = np.array(one_step)
        all_images.extend([one_step_arr])
    arr = np.concatenate(all_images, axis=1)
    print(arr.shape)
    arr = arr[:, : args.num_samples]
    print(arr.shape)
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
