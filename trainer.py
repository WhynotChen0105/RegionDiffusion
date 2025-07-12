import accelerate.utils
import torch
from ldm.util import instantiate_from_config
from utils.misc import AverageMeter, ProgressMeter, sec_2_hms, save_config
import time
from torch.utils.data import DataLoader
import os
import shutil
from utils.dist import get_rank
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from copy import deepcopy
from utils.checkpoint import save_ckpt, save_ckpt_and_result, ImageCaptionSaver, read_official_ckpt, \
    create_expt_folder_with_auto_resuming, get_trainable_parameters
from utils.optimizer import disable_grads, update_ema
from accelerate import Accelerator
class Trainer:
    def __init__(self, config):
        self.accelerator = Accelerator(gradient_accumulation_steps=config.accumulation_steps)
        self.config = config
        self.name, self.writer, checkpoint = create_expt_folder_with_auto_resuming(config.OUTPUT_ROOT, config.name)
        self.config_dict = save_config(config, self.name) if self.accelerator.is_main_process else None
        if self.accelerator.is_main_process:
            shutil.copyfile(config.yaml_file, os.path.join(self.name, "train_config_file.yaml"))
            self.config_dict = vars(config)
            torch.save(self.config_dict, os.path.join(self.name, "config_dict.pth"))

        # = = = = = = = = = = = = = = = = = create model and diffusion = = = = = = = = = = = = = = = = = #
        self.model = instantiate_from_config(config.model)
        self.autoencoder = instantiate_from_config(config.autoencoder)
        self.text_encoder = instantiate_from_config(config.text_encoder)
        self.diffusion = instantiate_from_config(config.diffusion)
        self.grounding_tokenizer_input = self.model.grounding_tokenizer_input
        # = = = = = = = = = = = = = = = = = = = = load from autoresuming ckpt or load from official ckpt = = = = = = = = = = = = = = = = = = = = #
        self.starting_iter = 0
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location='cpu')
            self.model.load_state_dict(checkpoint["model"])
            self.autoencoder.load_state_dict(checkpoint["autoencoder"])
            self.text_encoder.load_state_dict(checkpoint["text_encoder"])
            self.diffusion.load_state_dict(checkpoint["diffusion"])
            self.starting_iter = checkpoint["iters"]
            original_params_names = None
            if self.starting_iter >= config.total_iters:
                self.accelerator.wait_for_everyone()
                print("Training finished. Start exiting")
                exit()
        else:
            state_dict = read_official_ckpt(config.official_ckpt_name, 'cpu')
            # load original SD ckpt (with inuput conv may be modified)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict["model"], strict=False)
            assert unexpected_keys == []
            original_params_names = list(state_dict["model"].keys())  # used for sanity check later
            self.autoencoder.load_state_dict(state_dict["autoencoder"])
            self.text_encoder.load_state_dict(state_dict["text_encoder"], strict=False)
            self.diffusion.load_state_dict(state_dict["diffusion"])
            del state_dict
        self.autoencoder.eval()
        self.text_encoder.eval()
        disable_grads(self.autoencoder)
        disable_grads(self.text_encoder)


        # = = = = = = = = = = = = = = = = = create opt = = = = = = = = = = = = = = = = = #
        params = get_trainable_parameters(self.model, original_params_names)
        self.opt = torch.optim.AdamW(params, lr=config.base_learning_rate, weight_decay=config.weight_decay)
        self.model, self.autoencoder, self.text_encoder, self.diffusion = self.accelerator.prepare(self.model, self.autoencoder, self.text_encoder, self.diffusion)
        self.opt = self.accelerator.prepare(self.opt)
        if checkpoint is not None:
            self.opt.load_state_dict(checkpoint["opt"])

        #  = = = = = = = = = = = = = = = = = = = = = = EMA = = = = = = = = = = = = = = = = = = = = = = #
        if config.enable_ema:
            self.master_params = list(self.model.parameters())
            self.ema = deepcopy(self.model)
            self.ema_params = list(self.ema.parameters())
            self.ema.eval()
            self.ema = self.accelerator.prepare(self.ema)
            if checkpoint is not None:
                self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None
            self.ema_params = None



        # = = = = = = = = = = = = = = = = = = = = create scheduler = = = = = = = = = = = = = = = = = = = = #
        config.warmup_steps = config.warmup_steps * config.accumulation_steps
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps,
                                                             num_training_steps=config.total_iters)
        elif config.scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps)
        else:
            assert False
        self.scheduler = self.accelerator.prepare(self.scheduler)
        if checkpoint is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

            # = = = = = = = = = = = = = = = = = = = = create data = = = = = = = = = = = = = = = = = = = = #
        self.dataset_train = instantiate_from_config(config.dataset)
        loader_train = DataLoader(self.dataset_train, batch_size=config.batch_size, num_workers=config.workers, shuffle=True)
        self.loader_train = self.accelerator.prepare(loader_train)
        if self.accelerator.is_main_process:
            print("Total training images: ", self.dataset_train.total_images())

        # = = = = = = = = = = = = = = = = = = = = image_caption_saver = = = = = = = = = = = = = = = = = = = =#

        self.image_caption_saver = ImageCaptionSaver(self.name) if self.accelerator.is_main_process else None


    @torch.no_grad()
    def get_input(self, batch):
        z = self.autoencoder.encode(batch["image"])
        noise = torch.randn_like(z)
        context = self.text_encoder.encode(batch["caption"])
        _t = torch.rand(z.shape[0]).to(z.device)
        t = (torch.pow(_t, 1) * 1000).long()
        t = torch.where(t != 1000, t, 999)  # if 1000, then replace it with 999
        return z, noise, t, context

    def train_one_epoch(self, epoch, total_epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        progress = ProgressMeter(
            len(self.loader_train),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}|{}]".format(epoch + 1, total_epoch))

        end = time.time()

        self.model.train()
        for iter_idx, batch in enumerate(self.loader_train):
            data_time.update(time.time() - end)
            with self.accelerator.accumulate(self.model):
                # measure data loading time
                # forward
                self.opt.zero_grad(set_to_none=True)
                loss = self.run_one_step(batch)
                self.accelerator.backward(loss)
                self.opt.step()
                self.scheduler.step()
                if self.config.enable_ema:
                    update_ema(self.ema_params, self.master_params, self.config.ema_rate)
                losses.update(loss.item())
                if self.accelerator.is_main_process:
                    if (iter_idx % 10 == 0):
                        self.log_loss()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print progress
            print_freq = 10
            if iter_idx % print_freq == 0 and self.accelerator.is_main_process:
                secs = batch_time.avg * (self.config.total_iters - self.iter_idx)
                progress.display(iter_idx, lr=self.opt.param_groups[0]['lr'], remaining_time=sec_2_hms(int(secs)))

            self.iter_idx += 1

            # save ckpt as checkpoint_latest.pth every 2000 iters
            if self.iter_idx % 2000 == 0:
                save_ckpt(self.accelerator, self.config, self.model, self.text_encoder, self.autoencoder, self.opt, self.scheduler,
                          self.config_dict, self.diffusion, self.ema, self.iter_idx, self.name)
            # save ckpt and results every save_every_iters iters
            if self.iter_idx % self.config.save_every_iters == 0:
                save_ckpt_and_result(self.accelerator, self.config, self.model, self.text_encoder, self.autoencoder, self.opt,
                                     self.scheduler, self.config_dict, self.diffusion, self.ema, self.iter_idx,
                                     self.loader_train, self.dataset_train, self.grounding_tokenizer_input,
                                     self.image_caption_saver, self.name)

    def start_training(self):
        self.config.total_iters = self.config.total_epochs * len(self.loader_train)
        self.iter_idx = self.starting_iter
        start_epoch = self.starting_iter // len(self.loader_train)
        # training loop
        for epoch in range(start_epoch, self.config.total_epochs):
            # if self.accelerator.is_main_process:
            #     self.loader_train.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch, self.config.total_epochs)

        # save the final ckpt and result
        if get_rank() == 0:
            save_ckpt_and_result(self.accelerator, self.config, self.model, self.text_encoder, self.autoencoder, self.opt, self.scheduler,
                                 self.config_dict, self.diffusion, self.ema, self.iter_idx, self.loader_train,
                                 self.dataset_train, self.grounding_tokenizer_input, self.image_caption_saver,
                                 self.name)
        print("Model training is completed!!!")

    def run_one_step(self, batch):
        x_start, noise, t, context = self.get_input(batch)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)
        grounding_input = self.grounding_tokenizer_input.prepare(batch)
        input = dict(x=x_noisy,
                     timesteps=t,
                     context=context,
                     grounding_input=grounding_input)
        model_output = self.model(input)
        target = noise
        loss = torch.pow((model_output - target), 2)
        loss = loss * batch["loss_masks"] if self.config.foreground_loss_mode else loss
        loss = loss.mean()
        self.loss_dict = {"loss": loss.item()}
        return loss

    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(k, v, self.iter_idx + 1)  # we add 1 as the actual name


