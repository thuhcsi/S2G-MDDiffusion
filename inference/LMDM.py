import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
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

