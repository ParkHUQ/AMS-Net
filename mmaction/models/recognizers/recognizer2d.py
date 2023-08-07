import torch
from torch import nn

from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        #num_segs = imgs.shape[0] // batches
        num_segs = self.backbone.num_segments
        #print('num_segment', self.backbone.num_segments)

        losses = dict()

        x = self.extract_feat(imgs)

        if self.backbone_from == 'torchvision':
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x, labels.squeeze())   #[N,C,T,7,7]

            # x = x.squeeze(2)
            # num_segs = 1

            num_segs = x.size()[2]
            x = x.transpose(1, 2)
            x = x.reshape((-1, ) + x.shape[2:])   #add [N*num_segs, inchannel, 7, 7]

            #x = x.squeeze(2)
            losses.update(loss_aux)

            '''X_Front and X_Behind'''
            # print(len(x), x[0].size(), x[1].size())
            # x_front = []
            # x_behind = []
            # for each in x:
            #     x_f, x_be = each.split(split_size=8, dim=2)
            #     x_front.append(x_f)
            #     x_behind.append(x_be)
            #     # print(x_f.size())
            # x_front, loss_aux_front = self.neck(x_front, labels.squeeze())   #[N,C,T,7,7]
            # x_behind, loss_aux_behind = self.neck(x_behind, labels.squeeze())  # [N,C,T,7,7]
            # # print(loss_aux_front)
            # loss_aux = loss_aux_front
            # loss_aux['loss_aux'] = (loss_aux_front['loss_aux'] + loss_aux_behind['loss_aux'])/2
            # # print('losssssss', loss_aux)
            #
            # x = [x_front, x_behind]
            # x = torch.cat(x, dim=2)
            # # print('xsize', x.size())
            #
            # num_segs = x.size()[2]
            # x = x.transpose(1, 2)
            # x = x.reshape((-1, ) + x.shape[2:])   #add [N*num_segs, inchannel, 7, 7]
            #
            # #x = x.squeeze(2)
            # losses.update(loss_aux)
        ''''''''''''''''''

        cls_score = self.cls_head(x, num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        #print("batch", batches)
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        #num_segs = imgs.shape[0] // batches
        num_segs = self.backbone.num_segments

        x = self.extract_feat(imgs)

        if self.backbone_from == 'torchvision':
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            # x = x.squeeze(2)
            # num_segs = 1

            num_segs = x.size()[2]
            x = x.transpose(1, 2)
            x = x.reshape((-1,) + x.shape[2:])  # add [N*num_segs, inchannel, 7, 7]

            #print('do_test', x.size())

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        cls_score = self.cls_head(x, num_segs)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)

        return cls_score

    def _do_fcn_test(self, imgs):
        # [N, num_crops * num_segs, C, H, W] ->
        # [N * num_crops * num_segs, C, H, W]
        batches = imgs.shape[0]
        #print("before", imgs.shape)
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = self.test_cfg.get('num_segs', self.backbone.num_segments)
        # print('num_segs', num_segs)
        # print('batches', batches)
        # print('xxxxx',imgs.shape)

        if self.test_cfg.get('flip', False):
            imgs = torch.flip(imgs, [-1])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
        else:
            x = x.reshape((-1, num_segs) +
                          x.shape[1:]).transpose(1, 2).contiguous()

        #print('fcn_test', x.size())
        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        cls_score = self.cls_head(x, fcn_test=True)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        #print(cls_score.size(), 'batches', batches)
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        if self.test_cfg.get('fcn_test', False):
            # If specified, spatially fully-convolutional testing is performed
            return self._do_fcn_test(imgs).cpu().numpy()
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        outs = self.cls_head(x, num_segs)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)
