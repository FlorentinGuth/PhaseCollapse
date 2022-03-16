import torch
import torch.nn as nn

class LGM_logits(nn.Module):

    def __init__(self, n_space, nb_channels_in, alpha=0.1, nb_classes=1000, use_std=False, avg_ker_size=1,
                 avgpool=False, bottleneck=False, bottleneck_size=256, no_classifier_bn=False):
        super(LGM_logits, self).__init__()

        self.no_classifier_bn = no_classifier_bn
        if not no_classifier_bn:
            self.bn = nn.BatchNorm2d(nb_channels_in)

        self.avg_ker_size = avg_ker_size
        self.avgpool = avgpool
        if self.avgpool:
            n = 1
        elif self.avg_ker_size > 1:
            n = n_space - avg_ker_size + 1
        else:
            n = n_space

        in_planes = nb_channels_in * (n ** 2)

        self.use_bottleneck = bottleneck
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(in_planes, bottleneck_size)
            rep_size = bottleneck_size
        else:
            rep_size = in_planes

        self.centroids = nn.Parameter(nn.init.xavier_uniform_(torch.empty(rep_size, nb_classes))) # (C,K) C: number of channels, K number of classes
        self.inv_std = nn.Parameter(torch.ones(rep_size, nb_classes), requires_grad=use_std) # (C,K) C: number of channels, K number of classes
        self.nb_classes = nb_classes
        self.alpha = alpha

    def forward(self, x, target=None):

        if not self.no_classifier_bn:
            x = self.bn(x)
        if self.avgpool:
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        elif self.avg_ker_size > 1:
            x = nn.functional.avg_pool2d(x, self.avg_ker_size, stride=1)

        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (B,C) B: batch_size, C number of channels
        if self.use_bottleneck:
            x = self.bottleneck(x)
        inv_var = self.inv_std**2   # (C,K) C: number of channels, K number of classes
        xT_lambda_inv_x = torch.mm(x**2, inv_var)   # (B,K)
        xT_lambda_inv_mu = torch.mm(x, self.centroids * inv_var)    # (B,K)
        muT_lambda_iv_mu = (self.centroids**2 * inv_var).sum(dim=0, keepdim=True).expand(batch_size, self.nb_classes)   # (B,K)
        neg_sqr_dist = -0.5 * (xT_lambda_inv_x - 2.0 * xT_lambda_inv_mu + muT_lambda_iv_mu)     # (B,K)
        log_det = torch.log(self.inv_std.prod(dim=0, keepdim=True)).expand(batch_size, self.nb_classes)     # (B,K)

        if target is not None:  # margin implementation only for training
            likelihood_loss = torch.gather(-(neg_sqr_dist + log_det), dim=1, index=target.unsqueeze(-1)).sum()/batch_size #B then scalar
            one_hot_target = nn.functional.one_hot(target, num_classes=self.nb_classes)  # (B,K)
            margin_adjust = torch.ones(batch_size, self.nb_classes).cuda() + self.alpha*one_hot_target  # (B,K)
            neg_sqr_dist = margin_adjust * neg_sqr_dist  # (B,K)
            logits = neg_sqr_dist + log_det
            return logits, likelihood_loss

        else:
            logits = neg_sqr_dist + log_det
            return logits