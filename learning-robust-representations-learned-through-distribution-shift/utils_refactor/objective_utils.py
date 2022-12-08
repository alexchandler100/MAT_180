# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])

        # target_matches.shape = BOM
        # target_matches[b, o, m] = 1 means at sample b, object at index o corresponds to buffer at index m.
        # out_prob.shape = BNM
        # cost_class = -einsum("b n m, b o m -> b n o", out_prob, target_matches)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Final cost matrix
        C = self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["labels"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # i[0] matches with j[0], i[1] matches with j[1], and so on.
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class BufferRetrievalHungarianMatcher(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, targets):
        # outputs.shape = BNM
        #   outputs[b, n, m] >> 0 implies that at sample b, the prediction at output slot n is similar to
        #   the prototype vector at index m.
        # targets.shape = BOM
        #   targets[b, o, m] = 1 implies that at sample b, a representative of the true object at index o
        #   can be found in the sampled buffer at index m.

        # The first step is to obtain the cost matrix, of shape BNO.
        # cost_matrix.shape = BNO
        #   cost_matrix[b, n, o] large implies that prediction at output slot n does not want to
        #   be matched to the ground-truth object at index o. Conversely,
        #   cost_matrix[b, n, o] small implies that prediction at output slot n wants to
        #   be matched to the ground-truth object at index o.
        # By multiplying the targets with outputs and summing along the M dimension, we'll get a large value
        # if the predicted protoype at index n is similar to the right prototypes corresponding to the
        # ground truth object. This means it's a good match, and we want it to have a low cost, hence the negation.
        cost_matrix = -torch.einsum("bnm,bom->bno", outputs, targets)
        batch_size = cost_matrix.shape[0]
        # indices is a list of 2-tuples. Consider the first element, n, o = indices[0].
        indices = [linear_sum_assignment(cost_matrix[b].cpu()) for b in range(batch_size)]
        return [(torch.as_tensor(n, dtype=torch.int64), torch.as_tensor(o, dtype=torch.int64)) for n, o in indices]


class BufferRetrievalSetCriterion(nn.Module):
    def __init__(self, matcher):
        super(BufferRetrievalSetCriterion, self).__init__()
        self.matcher = matcher
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # outputs.shape = BNM
        #   outputs[b, n, m] >> 0 implies that at sample b, the prediction at output slot n is similar to
        #   the prototype vector at index m.
        # targets.shape = BOM
        #   targets[b, o, m] = 1 implies that at sample b, a representative of the true object at index o
        #   can be found in the sampled buffer at index m.
        # First, do the matching.
        # indices is a list of 2-tuples, and len(indices) = batch_size.
        indices = self.matcher(outputs, targets)
        ns, os = zip(*indices)
        # ns.shape = os.shape = B
        ns, os = torch.stack(ns), torch.stack(os)
        # Permute the outputs and targets such that they're aligned.
        # We want aligned_outputs[b, u, :] to match with aligned_targets[b, u, :],
        # where u is some index.
        aligned_outputs = outputs[torch.arange(outputs.shape[0], device=outputs.device)[:, None], ns]
        aligned_targets = targets[torch.arange(outputs.shape[0], device=outputs.device)[:, None], os]
        # aligned_outputs = outputs[ns]
        # aligned_targets = targets[os]
        # Next step is to evaluate the binary cross entropy with logits
        loss_value = self.loss_fn(aligned_outputs, aligned_targets)
        return dict(loss_bce=loss_value)


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses

    @staticmethod
    def target_transform(target):
        # target is a tensor of shape (batch_size, num_digits)
        return [{"labels": t} for t in target]


def test_vanilla():
    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=10, matcher=matcher, weight_dict=dict(labels=1.0),
                             losses=["labels"], eos_coef=1.0)
    batch_size = 3
    num_digits_per_image = 2

    labels = torch.randint(0, 10, (batch_size, num_digits_per_image))

    targets = [{"labels": l} for l in labels]
    predictions = {"pred_logits": torch.randn(batch_size, 4, 11)}

    loss = criterion(predictions, targets)


def test_retrieval():
    matcher = BufferRetrievalHungarianMatcher()
    criterion = BufferRetrievalSetCriterion(matcher=matcher)

    batch_size = 1
    num_true_objects = 5
    num_prediction_slots = 4
    num_proto = 10

    # targets:
    buffer_hits = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
                                ], dtype=torch.float32)
    buffer_hits = buffer_hits[None, :, :]
    # buffer_hits = torch.randn(batch_size, num_true_objects, num_proto).gt_(0.)
    # outputs:
    proto_hits = torch.randn(batch_size, num_prediction_slots, num_proto)

    loss = criterion(proto_hits, buffer_hits)


if __name__ == "__main__":
    test_retrieval()
