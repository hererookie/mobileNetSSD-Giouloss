import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from ..utils import box_utils


# from box_utils import match, log_sum_exp


# MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=DEVICE)
def match_gious(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    loc_t[idx] = box_utils.point_form(priors)
    # print("truths:", truths.shape)
    overlaps = box_utils.jaccard(
        truths,
        box_utils.point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]  # Shape: [num_priors,4]
    # print("labels.shape:", labels.shape)  # torch.Size([3000, 1])
    # conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc_t[idx] = matches  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        print(self.gamma, self.alpha)

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        # class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class GiouLoss(nn.Module):
    """
        This criterion is a implemenation of Giou Loss, which is proposed in 
        Generalized Intersection over Union Loss for: A Metric and A Loss for Bounding Box Regression.

            Loss(loc_p, loc_t) = 1-GIoU

        The losses are summed across observations for each minibatch.

        Args:
            size_sum(bool): By default, the losses are summed over observations for each minibatch.
                                However, if the field size_sum is set to False, the losses are
                                instead averaged for each minibatch.
            predmodel(Corner,Center): By default, the loc_p is the Corner shape like (x1,y1,x2,y2)
            The shape is [num_prior,4],and it's (x_1,y_1,x_2,y_2)
            loc_p: the predict of loc
            loc_t: the truth of boxes, it's (x_1,y_1,x_2,y_2)
            
    """

    def __init__(self, pred_mode='Center', size_sum=True, variances=None):
        super(GiouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances

    def forward(self, loc_p, loc_t, prior_data):

        # loss_l = self.gious(loc_p, loc_t, giou_priors[pos_idx].view(-1, 4))

        num = loc_p.shape[0]

        if self.pred_mode == 'Center':
            decoded_boxes = box_utils.decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p

        # loss = torch.tensor([1.0])
        
        gious = 1.0 - box_utils.bbox_overlaps_giou(decoded_boxes, loc_t)
        # print("gious:", gious)

        loss = torch.sum(gious)

        if self.size_sum:
            loss = loss
        else:
            loss = loss / num
        return 5 * loss

class MultiboxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)

        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].

    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, priors, num_class, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        super(MultiboxLoss, self).__init__()

        self.use_gpu = True

        self.num_classes = num_class

        self.priors = priors
        # print("......................................................priors.shape:", priors.shape)
        self.priors.to(device)
        self.threshold = iou_threshold
        self.background_label = 0
        self.encode_target = False
        self.use_prior_for_matching = True
        self.do_neg_mining = True
        self.negpos_ratio = neg_pos_ratio
        self.neg_overlap = iou_threshold
        self.variance = size_variance
        # self.focalloss = FocalLoss(self.num_classes,gamma=2,size_average = False)
        self.gious = GiouLoss(pred_mode='Center', size_sum=True, variances=self.variance)
        self.loss_c = 'CrossEntropy'
        self.loss_r = 'Giou'
        if self.loss_r != 'SmoothL1' or self.loss_r != 'Giou':
            assert Exception("THe loss_r is Error, loss name must be SmoothL1 or Giou")
        elif self.loss_c != 'CrossEntropy' or self.loss_c != 'FocalLoss':
            assert Exception("THe loss_c is Error, loss name must be CrossEntropy or FocalLoss")
        elif self.loss_r == 'Giou':
            match_gious(self.threshold, truths, defaults, self.variance, labels,
                        loc_t, conf_t, idx)

    # def forward(self, predictions, targets):
    def forward(self, conf_data, loc_data, targets):
        # forward(self, confidence, predicted_locations, labels, gt_locations):
        """Multibox Loss
        Args:
            labels:
            loc_data:
            conf_data:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

                confidence: torch.Size([24, 3000, 21])
                locations: torch.Size([24, 3000, 4])
                boxes: torch.Size([24, 3000, 4])
                labels: torch.Size([24, 3000])
                truths: torch.Size([24, 3000, 4])

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        # loc_data, conf_data, priors = predictions
        num = loc_data.shape[0]
        # print("num of batch:", num)

        # num_classes = conf_data.size(2)

        # print("*************num_classes:", self.num_classes)
        priors = self.priors[:loc_data.shape[1], :]

        num_priors = self.priors.shape[0]

        # print("*************num_priors:", num_priors)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)

        conf_t = torch.LongTensor(num, num_priors)
        loc_t.requires_grad = False
        conf_t.requires_grad = False
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            # print("truths.shape:", truths.shape)
            # print("labels.shape:", labels.shape)
            # print("truths:", truths)
            # print("labels:", labels)
            if self.loss_r == 'SmoothL1':
                box_utils.match(self.threshold, truths, defaults, self.variance, labels,
                                loc_t, conf_t, idx)
            elif self.loss_r == 'Giou':
                # print("-------Giou loss-------")
                # print("truths.shape:", truths.shape)
                # print("truths:", truths)
                # print("defaults:", defaults)
                # print("labels:", labels)
                # print("loc_t:", loc_t)
                # print("conf_t:", conf_t)
                # print("idx:", idx)
                match_gious(self.threshold, truths, defaults, self.variance, labels,
                            loc_t, conf_t, idx)
        if self.use_gpu:
           loc_t = loc_t.cuda()
           conf_t = conf_t.cuda()
        # wrap targets
        # loc_t = Variable(loc_t, requires_grad=True)
        # conf_t = Variable(conf_t, requires_grad=True)

        pos = conf_t > 0

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        loc_p = loc_data[pos_idx].view(-1, 4).cuda()

        loc_t = loc_t[pos_idx].view(-1, 4).cuda()

        if self.loss_r == 'SmoothL1':
            loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        elif self.loss_r == 'Giou':
            giou_priors = self.priors.data.unsqueeze(0).expand_as(loc_data)
            loss_l = self.gious(loc_p, loc_t, giou_priors[pos_idx].view(-1, 4).cuda())
        # print("loss_l:", loss_l)
        # Compute max conf across batch for hard negative mining

        if self.loss_c == "CrossEntropy":
            # print("***2-pos:", pos)
            batch_conf = conf_data.view(-1, self.num_classes)
            batch_conf = batch_conf.to(device="cuda")
            conf_t = conf_t.to(device="cuda")
            # print("batch_conf.shape:", batch_conf.shape)  # [96000, 2]
            # print("conf_t.shape:", conf_t.shape)  # [32, 3000]
            # print("box_utils.log_sum_exp(batch_conf):", box_utils.log_sum_exp(batch_conf).shape)  # [96000, 1]
            # print("conf_t.view(-1, 1).shape:", conf_t.view(-1, 1).shape)  # [96000,1]
            # print("conf_t.shape:", conf_t.shape)  # [32, 3000]

            index = conf_t.view(-1, 1).long()
            # print("index.shape:", index.shape)
            # data2 = torch.sum(index, dim=0)
            # print("data2_shape: ", data2.shape)
            # print("data2: ", data2)
            # print("batch_conf.gather(1, conf_t.view(-1, 1)):", (batch_conf.gather(1, index)).shape)
            # torch.set_printoptions(profile="full")
            # index = index.view(-1, 1000)
            # print("index", index)

            loss_c = box_utils.log_sum_exp(batch_conf) - batch_conf.gather(1, index)
            # print("1-pos.shape:", pos.shape)
            # print("1-loss_c.shape:", loss_c.shape)
            # print("************")
            # Hard Negative Mining
            loss_c = loss_c.view(num, -1)
            # print("2-pos.shape:", pos.shape)
            # print("2-loss_c.shape:", loss_c.shape)
            # print("****************11111111111******************")
            # print("-----2-pos:", pos)
            # print("*****************2222222222*****************")
            # print("2-loss_c:", loss_c)
            loss_c[pos] = 0
            # print("3-pos.shape:", pos.shape)
            # print("3-loss_c.shape:", loss_c.shape)
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = pos.unsqueeze(2).expand_as(conf_data)
            neg_idx = neg.unsqueeze(2).expand_as(conf_data)
            conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_t[(pos + neg).gt(0)]
            loss_c = F.cross_entropy(conf_p.to(device="cuda"), targets_weighted.to(device="cuda"), reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        elif self.loss_c == "FocalLoss":
            batch_conf = conf_data.view(-1, self.num_classes)
            loss_c = self.focalloss(batch_conf, conf_t)

        N = num_pos.data.sum().double()
        # print("N:", N)
        loss_l = loss_l.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
