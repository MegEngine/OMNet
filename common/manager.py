import os
import numpy as np
import megengine as mge
import megengine.distributed as dist
from collections import defaultdict
from termcolor import colored

from common import utils


class Manager():
    def __init__(self, model, optimizer, params, dataloaders, writer, logger, scheduler):
        # params status
        self.params = params
        self.optimizer = optimizer
        self.model = model
        self.dataloaders = dataloaders
        self.writer = writer
        self.logger = logger
        self.scheduler = scheduler

        # metric_rule should be either Descende or Ascende
        self.metric_rule = params.metric_rule

        self.epoch = 0
        self.step = 0

        # 越低越好
        if self.metric_rule == "Descende":
            self.best_val_score = 100
            self.best_test_score = 100
        # 越高越好
        elif self.metric_rule == "Ascende":
            self.best_val_score = 0
            self.best_test_score = 0

        self.cur_val_score = 0
        self.cur_test_score = 0

        # train status
        self.train_status = defaultdict(utils.AverageMeter)

        # val status
        self.val_status = defaultdict(utils.AverageMeter)

        # test status
        self.test_status = defaultdict(utils.AverageMeter)

        # model status
        self.loss_status = defaultdict(utils.AverageMeter)

    def update_step(self):
        self.step += 1

    def update_epoch(self):
        self.epoch += 1

    def update_loss_status(self, loss, split, bs=None):
        if split == "train":
            for k, v in loss.items():
                bs = self.params.train_batch_size
                self.loss_status[k].update(val=v.item(), num=bs)
        elif split == "val":
            for k, v in loss.items():
                self.loss_status[k].update(val=v.item(), num=bs)
        elif split == "test":
            for k, v in loss.items():
                self.loss_status[k].update(val=v.item(), num=bs)
        else:
            raise ValueError("Wrong eval type: {}".format(split))

    def update_metric_status(self, metrics, split, bs):
        if split == "val":
            for k, v in metrics.items():
                self.val_status[k].update(val=v.item(), num=bs)
                self.cur_val_score = self.val_status[self.params.major_metric].avg
        elif split == "test":
            for k, v in metrics.items():
                self.test_status[k].update(val=v.item(), num=bs)
                self.cur_test_score = self.test_status[self.params.major_metric].avg
        else:
            raise ValueError("Wrong eval type: {}".format(split))

    def summarize_metric_status(self, metrics, split):
        if split == "val":
            for k in metrics:
                if k.endswith('MSE'):
                    self.val_status[k[:-3] + 'RMSE'].set(val=np.sqrt(self.val_status[k].avg))
                else:
                    continue
        elif split == "test":
            for k in metrics:
                if k.endswith('MSE'):
                    self.test_status[k[:-3] + 'RMSE'].set(val=np.sqrt(self.test_status[k].avg))
                else:
                    continue
        else:
            raise ValueError("Wrong eval type: {}".format(split))

    def reset_loss_status(self):
        for k, v in self.loss_status.items():
            self.loss_status[k].reset()

    def reset_metric_status(self, split):
        if split == "val":
            for k, v in self.val_status.items():
                self.val_status[k].reset()
        elif split == "test":
            for k, v in self.test_status.items():
                self.test_status[k].reset()
        else:
            raise ValueError("Wrong eval type: {}".format(split))

    def print_train_info(self):
        exp_name = self.params.model_dir.split('/')[-1]
        print_str = "{} Epoch: {:4d}, lr={:.1E} ".format(exp_name, self.epoch, self.scheduler.get_lr()[0])
        print_str += "total loss: {:.3f}({:.3f})".format(self.loss_status['total'].val, self.loss_status['total'].avg)
        return print_str

    def print_metrics(self, split, title="Eval", color="red", only_best=True):
        if split == "val":
            metric_status = self.val_status
            is_best = self.cur_val_score < self.best_val_score
        elif split == "test":
            metric_status = self.test_status
            is_best = self.cur_test_score < self.best_test_score
        else:
            raise ValueError("Wrong split string: {}".format(split))
        print_str = " | ".join("{}: {:.3f}".format(k, v.avg) for k, v in metric_status.items())
        if only_best:
            if is_best:
                utils.master_logger(self.logger,
                                    colored("Best Epoch: {}, {} Results: {}".format(self.epoch, title, print_str), color, attrs=["bold"]),
                                    dist.get_rank() == 0)
        else:
            utils.master_logger(self.logger, colored("Epoch: {}, {} Results: {}".format(self.epoch, title, print_str),
                                                     color,
                                                     attrs=["bold"]),
                                dist.get_rank() == 0)

    def check_best_save_last_checkpoints(self, save_latest_freq=5, save_best_after=50):

        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
        }
        if "val" in self.dataloaders:
            state["best_val_score"] = self.best_val_score
        if "test" in self.dataloaders:
            state["best_test_score"] = self.best_test_score

        # save latest checkpoint
        if self.epoch % save_latest_freq == 0:
            latest_ckpt_name = os.path.join(self.params.model_dir, "model_latest.pth")
            mge.save(state, latest_ckpt_name)
            self.logger.info("Saved latest checkpoint to: {}".format(latest_ckpt_name))

        # save val latest metrics, and check if val is best checkpoints
        if "val" in self.dataloaders:
            val_latest_metrics_name = os.path.join(self.params.model_dir, "val_metrics_latest.json")
            utils.save_dict_to_json(self.val_status, val_latest_metrics_name)

            # 越低越好
            if self.metric_rule == "Descende":
                is_best = self.cur_val_score < self.best_val_score
            # 越高越好
            elif self.metric_rule == "Ascende":
                is_best = self.cur_val_score > self.best_val_score
            else:
                raise Exception("metric_rule should be either Descende or Ascende")

            if is_best:
                # save metrics
                self.best_val_score = self.cur_val_score
                best_metrics_name = os.path.join(self.params.model_dir, "val_metrics_best.json")
                utils.save_dict_to_json(self.val_status, best_metrics_name)
                self.logger.info("Current is val best, score={:.3g}".format(self.best_val_score))
                # save checkpoint
                if self.epoch > save_best_after:
                    best_ckpt_name = os.path.join(self.params.model_dir, "val_model_best.pth")
                    mge.save(state, best_ckpt_name)
                    self.logger.info("Saved val best checkpoint to: {}".format(best_ckpt_name))

        # save test latest metrics, and check if test is best checkpoints
        if "test" in self.dataloaders:
            test_latest_metrics_name = os.path.join(self.params.model_dir, "test_metrics_latest.json")
            utils.save_dict_to_json(self.test_status, test_latest_metrics_name)
            # lower is better
            if self.metric_rule == "Descende":
                is_best = self.cur_test_score < self.best_test_score
            # higher is better
            elif self.metric_rule == "Ascende":
                is_best = self.cur_test_score > self.best_test_score
            else:
                raise Exception("metric_rule should be either Descende or Ascende")
            if is_best:
                # save metrics
                self.best_test_score = self.cur_test_score
                best_metrics_name = os.path.join(self.params.model_dir, "test_metrics_best.json")
                utils.save_dict_to_json(self.test_status, best_metrics_name)
                self.logger.info("Current is test best, score={:.3g}".format(self.best_test_score))
                # save checkpoint
                if self.epoch > save_best_after:
                    best_ckpt_name = os.path.join(self.params.model_dir, "test_model_best.pth")
                    mge.save(state, best_ckpt_name)
                    self.logger.info("Saved test best checkpoint to: {}".format(best_ckpt_name))

    def load_checkpoints(self):
        state = mge.load(self.params.restore_file)
        ckpt_component = []
        if "state_dict" in state and self.model is not None:
            try:
                self.model.load_state_dict(state["state_dict"])

            except Warning("Using custom loading net"):
                net_dict = self.model.state_dict()
                if "module" not in list(state["state_dict"].keys())[0]:
                    state_dict = {"module." + k: v for k, v in state["state_dict"].items() if "module." + k in net_dict.keys()}
                else:
                    state_dict = {k: v for k, v in state["state_dict"].items() if k in net_dict.keys()}
                net_dict.update(state_dict)
                self.model.load_state_dict(net_dict, strict=False)
            ckpt_component.append("net")

        if not self.params.only_weights:
            if "optimizer" in state and self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(state["optimizer"])

                except Warning("Using custom loading optimizer"):
                    optimizer_dict = self.optimizer.state_dict()
                    state_dict = {k: v for k, v in state["optimizer"].items() if k in optimizer_dict.keys()}
                    optimizer_dict.update(state_dict)
                    self.optimizer.load_state_dict(optimizer_dict)
                ckpt_component.append("opt")

            if "scheduler" in state and self.train_status["scheduler"] is not None:
                try:
                    self.scheduler.load_state_dict(state["scheduler"])

                except Warning("Using custom loading scheduler"):
                    scheduler_dict = self.scheduler.state_dict()
                    state_dict = {k: v for k, v in state["scheduler"].items() if k in scheduler_dict.keys()}
                    scheduler_dict.update(state_dict)
                    self.scheduler.load_state_dict(scheduler_dict)
                ckpt_component.append("sch")

            if "step" in state:
                self.step = state["step"] + 1
                self.train_status["step"] = state["step"] + 1
                ckpt_component.append("step: {}".format(self.train_status["step"]))

            if "epoch" in state:
                self.epoch = state["epoch"] + 1
                self.train_status["epoch"] = state["epoch"] + 1
                ckpt_component.append("epoch: {}".format(self.train_status["epoch"]))

            if "best_val_score" in state:
                self.best_val_score = state["best_val_score"]
                ckpt_component.append("best val score: {:.3g}".format(self.best_val_score))

            if "best_test_score" in state:
                self.best_test_score = state["best_test_score"]
                ckpt_component.append("best test score: {:.3g}".format(self.best_test_score))

        ckpt_component = ", ".join(i for i in ckpt_component)
        utils.master_logger(self.logger, "Loaded models from: {}".format(self.params.restore_file), dist.get_rank() == 0)
        utils.master_logger(self.logger, "Ckpt load: {}".format(ckpt_component), dist.get_rank() == 0)
