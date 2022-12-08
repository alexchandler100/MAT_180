import torch
import torchvision
from einops import rearrange
import wandb

def get_bincount_of_keys_along_book_dim(key_bindings, args):
    bincount = torch.zeros((key_bindings.shape[1], args.num_pairs), dtype=torch.float)
    key_bindings = rearrange(key_bindings, " b c ... -> c (b ...)")
    for i in range(key_bindings.shape[0]):
        bincount[i] = torch.bincount(key_bindings[i].int(), minlength=args.num_pairs)
    plot = torchvision.utils.make_grid(bincount)
    image = wandb.Image(plot, caption="frequency bindings of keys per discrete code book")
    return image

def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def compute_batch_accuracy_with_buffer_matching(outputs):
    bs, num_obj, num_classes = outputs["pred_logits"].shape
    predicted_labels = torch.argmax(outputs["pred_logits"], dim=-1)
    target_bins = outputs["target_matches"].sum(dim=1)

    pred_bins = batched_bincount(predicted_labels, dim=1, max_value=num_classes)

    batch_accuracy_soft = (pred_bins == target_bins).masked_fill_((target_bins == 0), False).sum().float() / (bs*num_obj)
    batch_accuracy_hard = (torch.abs((pred_bins - target_bins)).sum(dim=-1) == 0).sum().float() / bs

    return batch_accuracy_soft, batch_accuracy_hard


def evaluate_batches_with_buffer_matching(dataloader, model, retrieval_dict=None, num_batches=10, multihot=False, target_criterion=None):
    device = next(model.parameters()).device
    model.eval()
    accuracy_soft = 0.0
    accuracy_hard = 0.0
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index >= num_batches:
                break
            inputs = batch[0].to(device)
            targets = list(map(lambda x: {"labels": x["labels"].to(device)}, batch[1]))
            output = model(inputs, retrieval_dict=retrieval_dict, target_labels=targets)
            batch_accuracy_soft, batch_accuracy_hard = compute_batch_accuracy_with_buffer_matching(output)
            accuracy_soft += 100 * float(batch_accuracy_soft.item())
            accuracy_hard += 100 * float(batch_accuracy_hard.item())

    bs, num_obj, num_classes = output["pred_logits"].shape
    predicted_labels = torch.argmax(output["pred_logits"], dim=-1)
    pred_bins = batched_bincount(predicted_labels, dim=1, max_value=num_classes)
    plots = torchvision.utils.make_grid(torch.cat((output["target_matches"].sum(dim=1)[0][None, :], pred_bins[0][None, :]), dim=0), nrow=2)

    return accuracy_soft / num_batches, accuracy_hard / num_batches, plots


def compute_batch_accuracy_hungarian_head(outputs, targets):
    bs, num_obj, num_classes = outputs["pred_logits"].shape
    predicted_labels = torch.argmax(outputs["pred_logits"], dim=-1)

    device = predicted_labels.device
    empty_class = num_classes - 1

    # target_labels = torch.stack([d["labels"] for d in targets])
    # no_targets = torch.full((bs, num_obj - target_labels.shape[-1]), num_classes-1, device=device)
    # target_labels = torch.cat([target_labels, no_targets], dim=1)

    target_labels = torch.stack([torch.cat([d["labels"],
                                            torch.full((num_obj - d["labels"].shape[-1],), empty_class, device=device)],
                                           dim=0)
                                 for d in targets])

    sorted_target_labels = torch.sort(target_labels, dim=-1)[0]
    sorted_predicted_labels = torch.sort(predicted_labels, dim=-1)[0]

    batch_accuracy = (sorted_predicted_labels == sorted_target_labels).sum().float() / (bs*num_obj)

    return batch_accuracy


def evaluate_batches(dataloader, model, retrieval_dict=None, num_batches=10, multihot=False, target_criterion=None):
    device = next(model.parameters()).device
    model.eval()
    accuracy = 0.0
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index >= num_batches:
                break
            inputs = batch[0].to(device)
            targets = list(map(lambda x: {"labels": x["labels"].to(device)}, batch[1]))
            output = model(inputs, retrieval_dict=retrieval_dict)

            if multihot:
                outputs = torch.sigmoid(output["pred_logits"])
                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5] = 0
                batch_accuracy = torch.all(outputs == targets, dim=1).float().mean()
            else:
                # if target_criterion is not None:
                #     targets = target_criterion.target_transform(targets)
                # batch_accuracy = torch.all(torch.sort(torch.argmax(output["pred_logits"], dim=-1))[0] ==
                #                            torch.sort(torch.stack([d["labels"] for d in targets]), dim=-1)[0],
                #                            dim=1).float().mean()
                batch_accuracy = compute_batch_accuracy_hungarian_head(output, targets)
            accuracy += 100 * float(batch_accuracy.item())

    return accuracy / num_batches
