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

def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model

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
        self.view_transform = A.Compose([
        A.RandomBrightnessContrast(p=1.0),
        A.Normalize(),
        ToTensorV2(),
    ])


    def run(self):
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

    def train_tta(self):
        train_loader = self.train_dataset.get_loader(epoch=0)
        for data_iter, batch in enumerate(train_loader):
            # Training and Adaptation
            self.training = True
            print(f'The batch: {len(batch)}')
            videos = batch[1]
            segment_loader = batch[2]
            batch = batch[0]
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
            print(f'The len of the batch: {len(batch)}')
            self.run_train(batch)

            # Predict Here
            self.model.training = False
            prediction, label = self.model.predict(videos, segment_loader)

            # save prediction 
            pred_npy = self.postprocess_save(prediction, label)
            save_path = 'tests.npy'
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
        
        targets = batch.masks

        print(f'batch size {batch.img_batch.shape, len(outputs), len(outputs[0])}')
        self.log_validation_image(outputs, targets, self.logging_conf.log_dir )

        # batch_size = len(batch.img_batch)

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
    
    def log_validation_image(self, output, target, savepath):
        savepaths = f'{savepath}/{self.epoch}'
        # target = target.squeeze(0)
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
                        pred_save = pred_save.squeeze(0)
                        
                        for idx in range(len(pred_save)):
                            save_image(pred_save[idx].float(), f'{savepaths}/pred_{idx}.png') 
                            save_image(target[frame_idx, idx].float(), f'{savepaths}/gt_{idx}.png') 

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
        
    def postprocess_save(self,prediction,label_shape):
        total_mask = np.zeros(label_shape)
        print(f'label shape {(total_mask.shape)}')
        for out_frame_idx in range(0, 160):
                for out_obj_id, out_mask in prediction[out_frame_idx].items():
                    print(f'The outmask shape{out_mask.shape}')
                    print(total_mask.shape)
                    total_mask[:,:, :, out_frame_idx] = out_mask
        return total_mask