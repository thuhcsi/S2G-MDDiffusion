import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import GestureDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GestureDiffusion
from model.model import GestureDecoder

import torch.distributed


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class LMDM:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        EMA=True,
        learning_rate=1e-4,
        weight_decay=0.02,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # torch.distributed.init_process_group(backend='gloo', init_method='env://') # for RTX 4090
        
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes

        self.repr_dim = repr_dim = 200

        feature_dim = 1024 + 35 if feature_type == "wavlm" else 35

        horizon_seconds = 3.2
        FPS = 25
        self.horizon = horizon = int(horizon_seconds * FPS)

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            print('load ckpt weight'+checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"]
            print(self.normalizer)

        model = GestureDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )

        diffusion = GestureDiffusion(
            model,
            horizon,
            repr_dim,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            print('load ckpt')
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        # load datasets
        print('load dataset')
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):  # already backuped pkl dataset
            print("load train dataset from pkl")
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            print("load test dataset from pkl")
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
            print('load dataset success')
        else: # no backuped pkl dataset
            print("load raw dataset")
            train_dataset = GestureDataset(
                feature_type = opt.feature_type,
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            )
            test_dataset = GestureDataset(
                feature_type = opt.feature_type,
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
            )
            # save dataset to pkl
            if self.accelerator.is_main_process:
                print("save dataset to pkl")
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"), protocol=4)
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"), protocol=4)

        # set normalizer
        self.normalizer = test_dataset.normalizer
        print(self.normalizer)

        # data loaders
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)

        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            print("init wandb")
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_ploss = 0
            avg_vloss = 0
            avg_aloss = 0

            # train
            self.train()
            for step, (x, cond_frame, cond, wavnames) in enumerate(
                load_loop(train_data_loader)
            ):
                loss, (pos_loss, v_loss, a_loss) = self.diffusion(
                    x, cond_frame, cond, t_override=None
                )
                self.optim.zero_grad()
                self.accelerator.backward(loss)

                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_ploss += pos_loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_aloss += a_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
            # Save model
            if (epoch % opt.save_interval) == 0:
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_ploss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_aloss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "Pos Loss": avg_ploss,
                        "V Loss": avg_vloss,
                        "A Loss": avg_aloss,
                    }
                    wandb.log(log_dict)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        if self.accelerator.is_main_process:
            wandb.run.finish()

    def render_sample(
            self, cond_frame, cond, last_half, mode
    ):
        render_count = 1
        shape = (render_count, self.horizon, self.repr_dim)
        cond_frame_input = cond_frame.unsqueeze(0).to(self.accelerator.device)
        cond_input = cond.unsqueeze(0).to(self.accelerator.device)
        last_half = last_half.unsqueeze(0).to(self.accelerator.device) if last_half is not None else None
        result = self.diffusion.render_sample(
            shape,
            cond_frame_input,
            cond_input,
            self.normalizer,
            epoch=None,
            render_out=None,
            last_half=last_half,
            mode=mode,
        )
        return result

