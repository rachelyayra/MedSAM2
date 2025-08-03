# import sys
# sys.path.append("/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/") 
from experiments.MedSAM_experiment import predict_and_save
# from datasets.datasets import MRIDataset, DRIVEDataset, STAREDataset
from torch.utils.data import DataLoader

import gc
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional
import copy

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

import albumentations as A
from albumentations.pytorch import ToTensorV2


from training.optimizer import construct_optimizer

from training.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)

from training.trainer import unwrap_ddp_if_wrapped

from training.utils.data_utils import BatchedVideoDatapoint, MRIDataset
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
    set_seeds,
    setup_distributed_backend,
)


from training.trainer import Trainer




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

        self.test_time = True

        if self.test_time:
            self.mode = 'test_time'

        self.view_transform = A.Compose([
        A.RandomBrightnessContrast(p=1.0),
        A.Normalize(),
        ToTensorV2()
    ])

    def run(self):
        assert self.mode in ["train", "train_only", "val", "test_time"]
        if self.mode == "train":
            if self.epoch > 0:
                logging.info(f"Resuming training from epoch: {self.epoch}")
                # resuming from a checkpoint
                if self.is_intermediate_val_epoch(self.epoch - 1):
                    logging.info("Running previous val epoch")
                    self.epoch -= 1
                    self.run_val()
                    self.epoch += 1
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        elif self.mode == "train_only":
            self.run_train()
        elif self.mode == "test_time":
            self.model = unwrap_ddp_if_wrapped(self.model)
            self.original_state = self.model.state_dict()
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


    def train_tta(self):
        train_loader = self.train_dataset.get_loader(epoch=0)
        for data_iter, batch in enumerate(train_loader):
            self.model.load_state_dict(self.original_state, strict=True)
            self.optim = construct_optimizer(
                self.model,
                self.optim_conf.optimizer,
                self.optim_conf.options,
                self.optim_conf.param_group_modifiers,
            )
            self.scaler = torch.cuda.amp.GradScaler()
            self.data_iter = data_iter
            self.epoch = 0
            self.run_train(batch)
            video_ids = batch.metadata.video_ids 
            unique_vids = torch.unique(video_ids)
            unique_vids_list = unique_vids.tolist() 
            # Save checkpoint before adaptation
            # self.save_checkpoint(self.epoch + 1)
            video_path =  "/scratch_net/ken/radjoe/BraTS/Validation/images_UNN"  # PATH to MOSE JPEGImages folder 
            label_path = "/scratch_net/ken/radjoe/BraTS/Validation/labels_UNN"
            all_files = [i for i in sorted(os.listdir(video_path)) if i.endswith('.npy')]

            data_names = [all_files[idx] for idx in unique_vids_list]

            dataset = MRIDataset(video_path, label_path, img_list = data_names)
            print(dataset)
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1
            print(f'This is the word size: {world_size}')
            checkpoint_paths = self.save_checkpoint(epoch=self.data_iter, rank=rank)
            pred_dataloader = DataLoader(dataset, shuffle=False)

            savepath = f'/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results_TTA/BraTS_GLI/MedSAM2/rank_{rank}'
            model_type = 'video'

            print(f'THE PREDICTOR {len(dataset), data_names } ')
            predict_and_save(checkpoint=checkpoint_paths[0], model_cfg='configs/sam2.1_hiera_t512.yaml',video_path=video_path, dataloader=pred_dataloader,savepath =savepath, model_type=model_type, )
            print('THE PREDICTOR ')
            os.remove(checkpoint_paths[0])

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
            enabled=self.optim_conf.amp.enabled,
            dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
        ):
            loss_dict, batch_size, extra_losses = self._step(
                batch,
                self.model,
                phase,
            )

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
        phase: str,
    ):

        results = self.generate_batch_views(batch=batch)
        
        
        outputs_batch = []
        outputs = model(batch)
        outputs_batch.append(outputs)
            
        for i in results:
            outputs = model(i)
            outputs_batch.append(outputs)

        batch_size = len(batch.img_batch)


        key = batch.dict_key  # key for dataset

        loss = self.loss[key](outputs_batch)
        
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



    def generate_batch_views(self, batch: BatchedVideoDatapoint, num_views=4):
            # batch_tensor: [B, C, H, W]
            results = []
            for i in range(num_views):
                new_batch = BatchedVideoDatapoint(
                img_batch=batch.img_batch,
                obj_to_frame_idx=batch.obj_to_frame_idx,
                masks=batch.masks,
                metadata= batch.metadata,
                dict_key=batch.dict_key,
                batch_size = batch.batch_size
                )
                for i in range(len(batch.img_batch[0])):

                    for j in range(len(batch.img_batch[:, 0])):

                        temp = torch.permute(batch.img_batch[j][i], (1, 2, 0))
                        temp = temp.cpu().numpy()
                        temp = self.view_transform(image = temp)['image']

                        new_batch.img_batch[j][i] = temp

                results.append(new_batch)
            
            return results

