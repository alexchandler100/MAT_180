import logging
from compo_r.utils_refactor import argparse_utils, util

import torch
from torch import nn
import wandb
from utils_refactor import argparse_utils, util
from models.models_perceiver import PerceiverRetriever
from utils_refactor import argparse_utils, util

logger = logging.getLogger(__name__)


def evaluate_model(model, val_loader, device, sampled_buffer_size, num_batches=10, buffer_data_model=None):
    h_loss = util.HLoss()

    def validation_step(batch, batch_idx):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        output = model(inputs, sampled_buffer_size=sampled_buffer_size, buffer_data_model=buffer_data_model)

        y_hat = output["prediction"]

        if "attn_reps" in output and "attn_retrieval" in output:
            entropy_attn_reps = h_loss(output["attn_reps"].reshape((-1, output["attn_reps"].shape[-1])))
            if output["attn_retrieval"] is not None:
                entropy_attn_retrieval = h_loss(
                    output["attn_retrieval"].reshape((-1, output["attn_retrieval"].shape[-1])))
            else:
                entropy_attn_retrieval = 1e-8
        else:
            entropy_attn_retrieval = 1e-8
            entropy_attn_reps = 1e-1

        num = y_hat.shape[0]

        outputs = torch.argmax(y_hat, dim=1)
        correct = (outputs == labels).sum().item()

        return num, correct, entropy_attn_reps, entropy_attn_retrieval

    total_num = 0
    total_correct = 0
    total_entropy_attn_reps = 0
    total_entropy_attn_retrieval = 0
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= num_batches:
            break
        num, correct, entropy_attn_reps, entropy_attn_retrieval = validation_step(batch, batch_idx)
        total_num += num
        total_correct += correct
        total_entropy_attn_reps += entropy_attn_reps
        total_entropy_attn_retrieval += entropy_attn_retrieval
    acc = total_correct / total_num
    return {"acc": acc,
            "entropy_attn_reps": total_entropy_attn_reps / total_num,
            "entropy_attn_retrieval": total_entropy_attn_retrieval / total_num}


def train(args):
    util.seed_everything(args.seed)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    dataloader_test, dataloader_train = util.vanilla_mnist_loaders(args)

    buffer_model = util.load_buffer_model(args)

    if args.buffer_path == "none":
        rep_buffer = None
    else:
        rep_buffer = torch.load(args.buffer_path, map_location=device)
        for key in rep_buffer:
            rep_buffer[key] = rep_buffer[key].to(device=device)
            rep_buffer[key] = rep_buffer[key].float()

    model = PerceiverRetriever(retrieval_buffer=rep_buffer,
                               num_freq_bands=args.num_freq_bands,
                               input_channels=3,
                               depth=args.depth,
                               max_freq=10,
                               num_classes=100,
                               num_latents=args.num_latents,
                               cross_dim_head=args.cross_dim_head,
                               cross_heads=args.cross_heads,
                               latent_heads=args.latent_heads,
                               latent_dim=args.latent_dim,
                               attn_dropout=args.attn_dropout,
                               ff_dropout=args.ff_dropout,
                               #batch=args.batch_size,
                               attn_retrieval=args.attn_retrieval,
                               retrieval_access=args.retrieval_access,
                               no_labels_from_reps=False)

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    inputs, labels = None, None
    if args.overfit_batch:
        batch = next(iter(dataloader_train))
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

    # Training loop
    train_step = 0
    for epoch in range(args.epochs):
        for index, batch in enumerate(dataloader_train):

            if train_step % args.eval_every_steps == 0:
                model.eval()
                val_loss = evaluate_model(model, dataloader_test, device,
                                          sampled_buffer_size=args.sampled_buffer_size,
                                          num_batches=args.eval_batches,
                                          buffer_data_model=buffer_model)
                wandb.log({"accuracy_test": val_loss["acc"],
                           "train_steps": train_step})
                util.save_model(args, model, train_step)
                model.train()

            train_step += 1
            if not args.overfit_batch:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

            output = model(inputs,
                           sampled_buffer_size=args.sampled_buffer_size,
                           buffer_data_model=buffer_model)
            loss = criterion(output["prediction"], labels)

            loss.backward()
            optimizer.step()

            num = output["prediction"].shape[0]

            outputs = torch.argmax(output["prediction"], dim=1)
            correct = (outputs == labels).sum().item()

            wandb.log({"train_loss": float(loss.item()),
                       "accuracy_train": float(correct) / num,
                       "train_steps": train_step})

            optimizer.zero_grad()

    util.save_model(args, model, train_step)


if __name__ == '__main__':
    run_args = argparse_utils.ArgumentParserWrapper().parse()
    logger.info("Model identifier: " + run_args.identifier)
    wandb.init(project="retrieval", name="test_run")
    wandb.config.update(run_args)
    wandb.run.name = f"perceiver_vanilla_mnist_{run_args.name}"
    train(run_args)
