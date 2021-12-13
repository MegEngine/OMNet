import argparse
import logging
import os

import dataset.data_loader as data_loader

import model.net as net

from common import utils
from loss.losses import compute_losses, compute_metrics
from common.manager import Manager
import megengine.distributed as dist
import megengine.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="experiments/base_model", help="Directory containing params.json")
parser.add_argument("--restore_file", default="best", help="name of the file in --model_dir containing weights to load")


def evaluate(model, manager):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    # set model to evaluation mode
    model.eval()

    # compute metrics over the dataset
    if manager.dataloaders["val"] is not None:
        # loss status and val status initial
        manager.reset_loss_status()
        manager.reset_metric_status("val")
        for data_batch in manager.dataloaders["val"]:
            # compute the real batch size
            bs = data_batch["points_src"].shape[0]
            # move to GPU if available
            data_batch = utils.tensor_mge(data_batch)
            # compute model output
            output_batch = model(data_batch)
            # compute all loss on this batch
            loss = compute_losses(data_batch, output_batch, manager.params)
            metrics = compute_metrics(data_batch, output_batch, manager.params)
            if world_size > 1:
                for k, v in loss.items():
                    loss[k] = F.distributed.all_reduce_sum(v) / world_size
                for k, v in metrics.items():
                    metrics[k] = F.distributed.all_reduce_sum(v) / world_size
            manager.update_loss_status(loss, "val", bs)
            # compute all metrics on this batch
            manager.update_metric_status(metrics, "val", bs)

        # update val data to tensorboard
        if rank == 0:
            # compute RMSE metrics
            manager.summarize_metric_status(metrics, "val")

            manager.writer.add_scalar("Loss/val", manager.loss_status["total"].avg, manager.epoch)
            # manager.logger.info("Loss/valid epoch {}: {:.4f}".format(manager.epoch, manager.loss_status["total"].avg))
            for k, v in manager.val_status.items():
                manager.writer.add_scalar("Metric/val/{}".format(k), v.avg, manager.epoch)
            # For each epoch, print the metric
            manager.print_metrics("val", title="Val", color="green")

    if manager.dataloaders["test"] is not None:
        # loss status and val status initial
        manager.reset_loss_status()
        manager.reset_metric_status("test")
        for data_batch in manager.dataloaders["test"]:
            # compute the real batch size
            bs = data_batch["points_src"].shape[0]
            # move to GPU if available
            data_batch = utils.tensor_mge(data_batch)
            # compute model output
            output_batch = model(data_batch)
            # compute all loss on this batch
            loss = compute_losses(data_batch, output_batch, manager.params)
            metrics = compute_metrics(data_batch, output_batch, manager.params)
            if world_size > 1:
                for k, v in loss.items():
                    loss[k] = F.distributed.all_reduce_sum(v) / world_size
                for k, v in metrics.items():
                    metrics[k] = F.distributed.all_reduce_sum(v) / world_size
            manager.update_loss_status(loss, "test", bs)
            # compute all metrics on this batch
            manager.update_metric_status(metrics, "test", bs)

        # update test data to tensorboard
        if rank == 0:
            # compute RMSE metrics
            manager.summarize_metric_status(metrics, "test")

            manager.writer.add_scalar("Loss/test", manager.loss_status["total"].avg, manager.epoch)
            # manager.logger.info("Loss/test epoch {}: {:.4f}".format(manager.epoch, manager.loss_status["total"].avg))
            for k, v in manager.val_status.items():
                manager.writer.add_scalar("Metric/test/{}".format(k), v.avg, manager.epoch)
            # For each epoch, print the metric
            manager.print_metrics("test", title="Test", color="red")


def test(model, manager):
    """Test the model with loading checkpoints.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    # set model to evaluation mode
    model.eval()

    # compute metrics over the dataset
    if manager.dataloaders["val"] is not None:
        # loss status and val status initial
        manager.reset_loss_status()
        manager.reset_metric_status("val")
        for data_batch in manager.dataloaders["val"]:
            # compute the real batch size
            bs = data_batch["points_src"].shape[0]
            # move to GPU if available
            data_batch = utils.tensor_mge(data_batch)
            # compute model output
            output_batch = model(data_batch)
            # compute all loss on this batch
            loss = compute_losses(data_batch, output_batch, manager.params)
            manager.update_loss_status(loss, "val", bs)
            # compute all metrics on this batch
            metrics = compute_metrics(data_batch, output_batch, manager.params)
            manager.update_metric_status(metrics, "val", bs)

        # compute RMSE metrics
        manager.summarize_metric_status(metrics, "val")
        # For each epoch, update and print the metric
        manager.print_metrics("val", title="Val", color="green")

    if manager.dataloaders["test"] is not None:
        # loss status and test status initial
        manager.reset_loss_status()
        manager.reset_metric_status("test")
        for data_batch in manager.dataloaders["test"]:
            # compute the real batch size
            bs = data_batch["points_src"].shape[0]
            # move to GPU if available
            data_batch = utils.tensor_mge(data_batch)
            # compute model output
            output_batch = model(data_batch)
            # compute all loss on this batch
            loss = compute_losses(data_batch, output_batch, manager.params)
            manager.update_loss_status(loss, "test", bs)
            # compute all metrics on this batch
            metrics = compute_metrics(data_batch, output_batch, manager.params)
            manager.update_metric_status(metrics, "test", bs)

        # compute RMSE metrics
        manager.summarize_metric_status(metrics, "test")
        # For each epoch, print the metric
        manager.print_metrics("test", title="Test", color="red")


if __name__ == "__main__":
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, "evaluate.log"))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    model = net.fetch_net(params)

    # Initial status for checkpoint manager
    manager = Manager(model=model, optimizer=None, scheduler=None, params=params, dataloaders=dataloaders, writer=None, logger=logger)

    # Reload weights from the saved file
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # Evaluate
    test(model, manager)
