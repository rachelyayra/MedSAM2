from pyexpat import model
from training.trainer import Trainer

import gc
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

import numpy as np
import torchvision.transforms.functional as F
import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.backends.cuda import sdp_kernel

from training.optimizer import construct_optimizer

from training.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)
from training.utils.data_utils import BatchedVideoDatapoint
from training.utils.distributed import all_reduce_max, barrier, get_rank

from training.utils.logger import Logger, setup_logging

from training.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    EarlyStopper,
    set_seeds,
    setup_distributed_backend,
)

from torchvision.utils import save_image

np.random.seed(42)

def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


torch.autograd.set_detect_anomaly(True)  # slow but precise

def watch_nan_grads(model):
    def hook(name):
        def _h(g):
            if g is None: return
            if not torch.isfinite(g).all():
                gg = torch.nan_to_num(g)
                print(f"[NaN grad @ {name}] min={float(gg.min())} max={float(gg.max())}")
        return _h

    for n,p in model.named_parameters():
        if p.requires_grad:
            p.register_hook(hook(f"param:{n}"))

class TestTimeTrainer(Trainer):
    def __init__(self, *, data, model, logging, checkpoint, max_epochs, **kwargs):
        super().__init__(
            data=data,
            model=model,
            logging=logging,
            checkpoint=checkpoint,
            max_epochs=max_epochs,
            **kwargs
        )

        self.brightness = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,      # uniform in [-0.2, +0.2]
                contrast_limit=0.2,        # uniform in [-0.2, +0.2]
                p=1.0                      # always apply, but randomized
            ),
            ToTensorV2(),
        ])

        # Tiny blur (random kernel, random sigma)
        self.gaussian_blur = A.Compose([
            A.GaussianBlur(
                blur_limit=(3, 5),         # random 3x3 or 5x5 kernel
                sigma_limit=(0.1, 0.8),    # mild blur
                p=1.0
            ),
            ToTensorV2(),
        ])

        # Light Gaussian noise
        self.gaussian_noise = A.Compose([
            A.GaussNoise(
                var_limit=(2e-4, 1e-2),    # std ≈ 0.01–0.07 if data in [0,1]
                mean=0.0,
                p=1.0
            ),
            ToTensorV2(),
        ])

        self.randomgamma = A.Compose([
            A.RandomGamma(
            gamma_limit=(85, 115), p=1.0
            ),
            ToTensorV2(),
        ])  

        self.hflip = A.Compose([
            A.HorizontalFlip(p=1.0),
            ToTensorV2(),
        ])
        self.vflip = A.Compose([
            A.VerticalFlip(p=1.0),
            ToTensorV2(),
        ])
        self.norm = A.Compose([A.Normalize(mean=(0.485,0.456,0.406),
                              std=(0.229,0.224,0.225)),
                  ToTensorV2()])

        self.transforms = [self.brightness , self.gaussian_blur, self.gaussian_noise]

        self.geo_transforms = [self.hflip]

        self.ema_decay = 0.999   # try 0.996–0.9997
        self.warmup_frac = 0.3   # for your consistency ramp, optional

    @torch.no_grad()
    def ema_update(self, teacher: torch.nn.Module,
                student: torch.nn.Module,
                decay: float):
        # parameters
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)
        # BatchNorm buffers (running_mean/var, num_batches_tracked)
        for tb, sb in zip(teacher.buffers(), student.buffers()):
            tb.data.copy_(sb.data)

    def run(self):
        self.model = unwrap_ddp_if_wrapped(self.model)
        self.original_state = copy.deepcopy(self.model.state_dict())
        self.train_tta()

    def run_train(self, batch):
        
        while self.epoch < self.max_epochs:
            barrier()
            outs = self.train_epoch(batch)
            outs['Train Epoch'] = self.epoch
            outs['Batch Number'] = self.data_iter
        
            self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

            # log train to text file.
            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")

            self.epoch += 1

    def train_tta(self):
        train_loader = self.train_dataset.get_loader(epoch=0)
        print(f'The length of the train loader: {len(train_loader)}')
        for data_iter, batch in enumerate(train_loader):
            # Training and Adaptation
            self.training = True
            print(f'The batch: {len(batch)}')
            videos = batch[1]
            segment_loader = batch[2]
            batch = batch[0]
            self.model.load_state_dict(self.original_state, strict=True)
            self.teacher = copy.deepcopy(self.model)
            self.optim = construct_optimizer(
                self.model,
                self.optim_conf.optimizer,
                self.optim_conf.options,
                self.optim_conf.param_group_modifiers,
            )
            self.scaler = torch.cuda.amp.GradScaler()
            self.data_iter = data_iter
            self.epoch = 0
            print(f'The len of the batch: {len(batch)}')
            self.run_train(batch)

            # Predict Here
            self.model.training = False
            prediction, label = self.model.predict(videos, segment_loader)

            # save prediction 
            pred_npy = self.postprocess_save(prediction, label)
            save_path = f'{self.logging_conf.log_dir}/BraTS_GLI'
            os.makedirs(save_path, exist_ok=True)
            save_path = f'{save_path}/{videos.video_name[:-8]}.npy'
            np.save(save_path, pred_npy)

            

    def train_epoch(self, batch_load):

        # Init stat meters
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN

        iters_per_epoch = len(batch_load)

        loss_names = []
        for batch_key in self.loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{},{}]".format(self.epoch, self.data_iter),
        )

        # Model training loop
        self.model.train()
        end = time.time()


        batch = batch_load.to(
                self.device, non_blocking=True
            )  # move tensors in a tensorclass

        try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)


                # Clipping gradients and detecting diverging gradients
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self.gradient_clipper(model=self.model)

                if self.gradient_logger is not None:
                    self.gradient_logger(
                        self.model, rank=self.distributed_rank, where=self.where
                    )

                # Optimizer step: the scaler will make sure gradients are not
                # applied if the gradients are infinite
                self.scaler.step(self.optim.optimizer)
                self.scaler.update()
                self.ema_update(self.teacher, self.model, self.ema_decay)

                # measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                mem_meter.update(reset_peak_usage=True)


            # Catching NaN/Inf errors in the loss
        except FloatingPointError as e:
                raise e

        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        self._log_sync_data_times(Phase.TRAIN, data_times)

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])

        return out_dict




    def _run_step(
        self,
        batch: BatchedVideoDatapoint,
        phase: str,
        loss_mts: Dict[str, AverageMeter],
        extra_loss_mts: Dict[str, AverageMeter],
        raise_on_error: bool = True,
    ):
        """
        Run the forward / backward
        """

        # it's important to set grads to None, especially with Adam since 0
        # grads will also update a model even if the step doesn't produce
        # gradients
        self.optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            enabled=False,
            # dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
        ), sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
            loss_dict, batch_size, extra_losses = self._step(
                batch,
                self.model,
                self.teacher,
                phase,
            )
        print(f'This is the losses{loss_dict.keys()}')
        assert len(loss_dict) == 1
        loss_key, loss = loss_dict.popitem()

        if not math.isfinite(loss.item()):
            error_msg = f"Loss is {loss.item()}, attempting to stop training"
            logging.error(error_msg)
            if raise_on_error:
                raise FloatingPointError(error_msg)
            else:
                return
        
        

        self.scaler.scale(loss).backward()
        

        loss_mts[loss_key].update(loss.item(), batch_size)
        for extra_loss_key, extra_loss in extra_losses.items():
            if extra_loss_key not in extra_loss_mts:
                extra_loss_mts[extra_loss_key] = AverageMeter(
                    extra_loss_key, self.device, ":.2e"
                )

            extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size)


    def _step(
        self,
        batch: BatchedVideoDatapoint,
        model: nn.Module,
        teacher: nn.Module,
        phase: str,
    ):  
        teacher.eval()
    #     feats = {}
    #     h_enc = model.image_encoder.register_forward_hook(
    #     lambda m,i,o: feats.setdefault("enc", o.detach())
    # )
        print(f'Checking for generator')
        results = self.generate_batch_views(batch=batch, transforms =self.transforms)
        batch = self.norm_views(batch)[0]
        print(f'After Checking for generator')
        targets = batch.masks
        outputs_batch = []
        print(f'batch information: {batch}')
        outputs = model(batch)  
        outputs_batch.append(outputs)


        with torch.no_grad():
            outputs_anchor = teacher(batch)


            outputs_batch.append(outputs_anchor)


       
        for i in range(len(results)):
            aug_outs = model(results[i])

            self.log_validation_image(aug_outs, targets, self.logging_conf.log_dir, i )
            outputs_batch.append(aug_outs)

        batch_size = len(batch.img_batch)





        # compute_class_separation(encoder_feats, decoder_feats, targets, self.epoch)
        self.log_validation_image(outputs, targets, self.logging_conf.log_dir, 'outs' )

        print(f'batch size {batch.img_batch.shape, len(outputs), len(outputs[0])} and shape of targets: {targets.shape}')
        
        # decoder_feats = 

        batch_size = len(batch.img_batch)

        key = batch.dict_key  # key for dataset

        loss = self.loss[key](outputs_batch)



        params = [p for p in self.model.parameters() if p.requires_grad]
        
        Loss_1 = loss["loss_mask"]
        Loss_2 = loss["loss_dice"]
        Loss_3 = loss["loss_iou"]

        print(f'This is the losses{Loss_1} and {Loss_2} and {Loss_3}')
        print(f'Requires grad: Loss1 {Loss_1.requires_grad}, Loss2 {Loss_2.requires_grad}, Loss3 {Loss_3.requires_grad}')

                # print(f'This is the losses{losses.items()}')
        with torch.cuda.amp.autocast(enabled=False):
            g1 = grad_vec(Loss_1, params)
            g2 = grad_vec(Loss_2, params)
            # g3 = grad_vec(Loss_3, params)

        n1 = g1.norm()
        n2 = g2.norm()
        # n3 = g3.norm()
        cos = (g1 @ g2) / (n1.clamp_min(1e-12) * n2.clamp_min(1e-12))
        # print(f"||∇L1||={float(n1):.3e}  ||∇L2||={float(n2):.3e}  ||∇L3||={float(n3):.3e}  cos={float(cos):.4f}")
        
        
        loss_str = f"Losses/{phase}_{key}_loss"

        loss_log_str = os.path.join("Step_Losses", loss_str)



        # loss contains multiple sub-components we wish to log
        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                loss,
                self.steps[phase],
            )

        self.steps[phase] += 1

        ret_tuple = {loss_str: loss}, batch_size, step_losses


        if phase in self.meters and key in self.meters[phase]:
            meters_dict = self.meters[phase][key]
            if meters_dict is not None:
                for _, meter in meters_dict.items():
                    meter.update(
                        find_stages=outputs,
                        find_metadatas=batch.metadata,
                    )


        return ret_tuple
    
    def save_checkpoint(self, epoch, rank=0, checkpoint_names=None):
        checkpoint_folder = self.checkpoint_conf.save_dir
        os.makedirs(checkpoint_folder, exist_ok=True)
        if checkpoint_names is None:
            checkpoint_names = [f"checkpoint_rank{rank}"]
            if (
                self.checkpoint_conf.save_freq > 0
                and (int(epoch) % self.checkpoint_conf.save_freq == 0)
            ) or int(epoch) in self.checkpoint_conf.save_list:
                checkpoint_names.append(f"checkpoint_rank{rank}_{int(epoch)}")

        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

            checkpoint_paths = []
            for ckpt_name in checkpoint_names:
                checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

            state_dict = self.model.state_dict()
            state_dict = exclude_params_matching_unix_pattern(
                patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
            )

            checkpoint = {
                "model": state_dict,
                "optimizer": self.optim.optimizer.state_dict(),
                "epoch": epoch,
                "loss": self.loss.state_dict(),
                "steps": self.steps,
                "time_elapsed": self.time_elapsed_meter.val,
                "best_meter_values": self.best_meter_values,
            }
            if self.optim_conf.amp.enabled:
                checkpoint["scaler"] = self.scaler.state_dict()

            for checkpoint_path in checkpoint_paths:
                self._save_checkpoint(checkpoint, checkpoint_path)

            return checkpoint_paths



    def norm_views(self, batch: BatchedVideoDatapoint):
            # batch_tensor: [B, C, H, W]
        results = []

        img_batch_clone = batch.img_batch.clone()

            # Apply view transforms
        B, T, C, H, W = img_batch_clone.shape
        # consave = f'{self.logging_conf.log_dir}/before'
        # conafter = f'{self.logging_conf.log_dir}/after'
        # os.makedirs(consave, exist_ok=True)
        # os.makedirs(conafter, exist_ok=True)
        for j in range(B):
                for i in range(T):
                    frame = img_batch_clone[j, i]  # [C, H, W]
                    # save_image(frame, f'{consave}/{i}.png')

                    transformed = F.normalize(frame, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    # save_image(transformed, f'{conafter}/{i}.png')
                    # transformed_tensor = transformed.permute(0, 1, 2).contiguous() 
                    
                    img_batch_clone[j, i] = transformed

        new_batch = BatchedVideoDatapoint(
                img_batch=img_batch_clone,
                obj_to_frame_idx=copy.deepcopy(batch.obj_to_frame_idx),
                masks=copy.deepcopy(batch.masks),
                metadata=copy.deepcopy(batch.metadata),
                dict_key=batch.dict_key,
                batch_size=batch.batch_size
            )

        results.append(new_batch)

        return results

    
    def log_validation_image(self, output, target, savepath, con):
        savepaths = f'{savepath}/{con}/{self.epoch}/'
        os.makedirs(savepaths, exist_ok=True)
        print(f'Check unique: {target.unique(), target.shape}')
        targets_onehot = nn.functional.one_hot(target.long(), 4)
        print(f'Check shape of one_hot: {targets_onehot.shape}')
        targets_onehot = targets_onehot.permute(0, 1, 4, 2, 3).float()
        
        # targets_onehot = targets_onehot[:, 1:, :, :]  # drop channel 0
        print(f'Check shape: {targets_onehot.shape}')
        
        os.makedirs(savepaths,exist_ok=True)
        for frame_idx in range(len(output)):
            print(f'This is the frame idx: {frame_idx}')
            if frame_idx == 0:
                
                step_list = output[frame_idx]['multistep_pred_multimasks_high_res']

                print(f'List of steps: {len(step_list)}')
                for step_idx in range(len(step_list)):
                    if step_idx == 0:
                        pred_save = step_list[step_idx]
                        print(f'prediction saving: {pred_save.shape, target.shape}')
                        # pred_save = pred_save.squeeze(0)
                        
                        for idx in range(len(pred_save)):
                            
                            for clx in range(len(pred_save[idx])):
                                color_target = colorize_mask(target[frame_idx,idx].cpu())
                                save_image(color_target.float(), f'{savepaths}/gt_class{idx}.png')
                                save_image(targets_onehot[frame_idx, idx,clx ].float(), f'{savepaths}/gt_class_{idx}_{clx}.png')
                                save_image(pred_save[idx, clx].float(), f'{savepaths}/pred_class_{idx}_{clx}.png') 
                                print(f'Check shape: {target[frame_idx, idx, clx ].shape}')

    def generate_batch_views(self, batch: BatchedVideoDatapoint, transforms):
        results = []
        for trans in range(len(transforms)):
            print(f'transforms : {trans}')
            # Clone the tensor to ensure independence
            img_batch_clone = batch.img_batch.clone()

            # Apply view transforms
            B, T, C, H, W = img_batch_clone.shape
            consave = f'{self.logging_conf.log_dir}/before/{trans}'
            conafter = f'{self.logging_conf.log_dir}/after/{trans}'
            os.makedirs(consave, exist_ok=True)
            os.makedirs(conafter, exist_ok=True)
            for j in range(B):
                for i in range(T):
                    frame = img_batch_clone[j, i]  # [C, H, W]
                    save_image(frame, f'{consave}/{j}_{i}.png')
                    frame_np = frame.permute(1, 2, 0).cpu().numpy()  # [H, W, C]

                    transformed = transforms[trans](image=frame_np)['image']  # [H, W, C]
                    save_image(transformed, f'{conafter}/{j}_{i}.png')
                    transformed_tensor = transformed.permute(0, 1, 2).contiguous() 
                    transformed_tensor = F.normalize(transformed_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    img_batch_clone[j, i] = transformed_tensor

            new_batch = BatchedVideoDatapoint(
                img_batch=img_batch_clone,
                obj_to_frame_idx=copy.deepcopy(batch.obj_to_frame_idx),
                masks=copy.deepcopy(batch.masks),
                metadata=copy.deepcopy(batch.metadata),
                dict_key=batch.dict_key,
                batch_size=batch.batch_size
            )

            results.append(new_batch)

        return results
        
    def postprocess_save(self,prediction,label_shape):
        total_mask = np.zeros(label_shape)
        print(f'label shape {(total_mask.shape)}')
        for out_frame_idx in range(0, 160):
                for out_obj_id, out_mask in prediction[out_frame_idx].items():
                    print(f'The outmask shape{out_mask.shape}')
                    print(total_mask.shape)
                    total_mask[:,:, :, out_frame_idx] = out_mask
        return total_mask
    
    def grab_image_encodings(self):
        return self.model.image_encoder
    
    def compute_class_separation(self,
        encoder_feats,
        decoder_feats,
        labels,
        epoch,
        pca_enc,
        pca_dec,
        ):
        """
        Compute class separation on SAM2 features.

        Steps:
        2) Downsample labels to feature-map size (NEAREST).
        3) Flatten features to (N,C) and L2-normalize; build (N,) labels.
        4) Class-balanced sample (per_class_max).
        5) Compute separation metrics:
            - Between-class uniformity (↓ better)
            - Prototype margin (↑ better), off-diag cosine (↓)
            - Fisher ratio (↑), silhouette (↑)
        6) (Optional) PCA/UMAP for visualization and save plots.
        7) Save a CSV row with metrics; return metrics dict(s).
        """
        print(f'encoder_feats shape: {encoder_feats.shape}')
        print(f'decoder_feats shape: {decoder_feats.shape}')
        print(f'labels shape: {labels.shape}')

        # Downsample labels 

        labels_enc = nn.functional.interpolate(
        labels.float(),
        size=(32, 32),   # or use scale_factor=0.5
        mode="nearest"
        ).squeeze(1).long()  

        labels_dec = nn.functional.interpolate(
        labels.float(),
        size=(128, 128),   # or use scale_factor=0.5
        mode="nearest"
        ).squeeze(1).long()  
        print(f'Downsampled labels shape: {labels_enc.shape}') 
        print(f'Downsampled labels shape: {labels_dec.shape}')

        # Pca on features and plot
        # B, C, H, W = encoder_feats.shape

        encoder_feats_pca = pca_enc.transform(encoder_feats)  # [B*H*W, 2]

        # Back to [B, 2, H, W]
        encoder_feats_pca = torch.from_numpy(encoder_feats_pca).to(encoder_feats.device)
        # encoder_feats_pca = encoder_feats_pca.view(B, H, W, 2).permute(0, 3, 1, 2)

        print(f'PCA reduced encoder features shape: {encoder_feats_pca.shape}')

        # Select 

        vecs, labs = gather_class_pixels(encoder_feats_pca, labels_enc, ignore_index=None)
        print(f'PCA reduced encoder features shape: {vecs.shape, labs.shape}')

        plt.figure(figsize=(6,6))
        scatter = plt.scatter(vecs[:,0], vecs[:,1], c=labs.numpy(), cmap="tab10", s=1, alpha=0.5)
        plt.colorbar(scatter, ticks=range(labs.max().item()+1), label="class id")
        plt.title("Per-pixel encoder vectors projected by PCA")
        plt.savefig(f"encoder_vectors_pca_epoch_con_{epoch}.png", dpi=300, bbox_inches="tight")
        plt.show()

        # B, C, H, W = decoder_feats.shape
        # decoder_feats_flat = decoder_feats.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

        # Fit PCA on the channel dimension
        
        decoder_feats_pca = pca_dec.transform(decoder_feats)  # [B*H*W, 2]

        # Back to [B, 2, H, W]
        decoder_feats_pca = torch.from_numpy(decoder_feats_pca).to(decoder_feats.device)
        # decoder_feats_pca = decoder_feats_pca.view(B, H, W, 2).permute(0, 3, 1, 2)

        print(f'PCA reduced decoder features shape: {decoder_feats_pca.shape}')

        vecs, labs = gather_class_pixels(decoder_feats_pca, labels_dec[-1], ignore_index=None)

        print(f'PCA reduced decoder features shape: {vecs.shape, labs.shape, labs.unique()}')
        plt.figure(figsize=(6,6))
        scatter = plt.scatter(vecs[:,0], vecs[:,1], c=labs.numpy(), cmap="tab10", s=1, alpha=0.5)
        plt.colorbar(scatter, ticks=range(labs.max().item()+1), label="class id")
        plt.title("Per-pixel decoder vectors projected by PCA")
        plt.savefig(f"decoder_vectors_pca_epoch_con_{epoch}.png", dpi=300, bbox_inches="tight")
        plt.show()

        intra = intra_class_distance(decoder_feats_pca, labs, 4)
        inter = inter_class_distance(decoder_feats_pca, labs, 4)
        ratio = inter / (intra + 1e-6)
        print(f"Intra: {intra:.4f}, Inter: {inter:.4f}, Ratio: {ratio:.4f}")
        log_distances_to_txt(intra, inter, ratio, path=f"{self.logging_conf.log_dir}/distances_con.txt", step=epoch)
        return
    

    def full_validation(self):
        pass
    
def colorize_mask(mask, palette=None):
    """mask: [H,W] int tensor with class IDs"""
    if palette is None:
        # simple fixed palette for 4 classes
        palette = torch.tensor([
            [0,0,0],      # class 0 -> black
            [255,0,0],    # class 1 -> red
            [0,255,0],    # class 2 -> green
            [0,0,255],    # class 3 -> blue
        ], dtype=torch.uint8)

    print(f'colorize mask shape; {mask.shape}')
    mask = mask.squeeze(0)
    h,w = mask.shape
    mask_rgb = palette[mask.flatten()].view(h,w,3).permute(2,0,1)  # [3,H,W]
    return mask_rgb


def grad_vec(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    vec = [g.detach().flatten() for g in gs if g is not None]
    return torch.cat(vec) if vec else None




def gather_class_pixels(feats, labels, ignore_index=None):


    labs = labels.reshape(-1)                     # [B*H*W]
    if ignore_index is not None:
        mask = labs != ignore_index
        feats = feats[mask]
        labs = labs[mask]
    return feats, labs.cpu()

def feats_pixels_as_samples(encoder_feats: torch.Tensor):

    with torch.no_grad():
        feats = encoder_feats.detach().float().cpu()     # move to CPU
    N, C, H, W = feats.shape
    # pixels as samples, channels as features
    X = feats.permute(0, 2, 3, 1).reshape(N*H*W, C).numpy()  # [N*H*W, C]
    return X, (N, H, W)


def intra_class_distance(f_flat, y_flat, num_classes):
    dists = []
    for c in range(1,num_classes):
        mask = (y_flat == c)
        if mask.sum() > 1:
            f_c = f_flat[mask]
            centroid = f_c.mean(dim=0, keepdim=True)   # (1,C)
            dist = (f_c - centroid).pow(2).sum(dim=1).mean()
            dists.append(dist.item())
    return sum(dists) / len(dists)


def inter_class_distance(f_flat, y_flat, num_classes):
    centroids = []
    for c in range(1, num_classes):
        mask = (y_flat == c)
        if mask.sum() > 0:
            f_c = f_flat[mask]
            centroids.append(f_c.mean(dim=0))
    centroids = torch.stack(centroids)   # (K,C)
    # pairwise squared distances
    dmat = torch.cdist(centroids, centroids, p=2)  # (K,K)
    return dmat.mean().item()


def log_distances_to_txt(intra, inter, ratio, path="distances.txt", step=None):
    with open(path, "a") as f:
        if step is not None:
            f.write(f"Step {step}: Intra={intra:.4f}, Inter={inter:.4f}, Ratio={ratio:.4f}\n")
        else:
            f.write(f"Intra={intra:.4f}, Inter={inter:.4f}, Ratio={ratio:.4f}\n")