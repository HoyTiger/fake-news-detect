# -*- coding:utf-8 -*-
# author : Han
# date : 2021/1/18 18:51
# IDE : PyCharm
# FILE : vgg.py

import paddle.fluid as fluid


class ConvBlock(fluid.dygraph.Layer):
    """
    卷积+池化
    """

    def __init__(self, name_scope, num_channels, num_filters, groups):
        """构造函数"""
        super(ConvBlock, self).__init__(name_scope)

        self._conv2d_list = []
        init_num_channels = num_channels
        for i in range(groups):
            conv2d = self.add_sublayer(
                'bb_%d' % i,
                fluid.dygraph.Conv2D(
                    init_num_channels, num_filters=num_filters, filter_size=3,
                    stride=1, padding=1, act='relu'
                )
            )
            self._conv2d_list.append(conv2d)
            init_num_channels = num_filters

        self._pool = fluid.dygraph.Pool2D(
            pool_size=2, pool_type='max', pool_stride=2
        )

    def forward(self, inputs):
        """前向计算"""
        x = inputs
        for conv in self._conv2d_list:
            x = conv(x)
        x = self._pool(x)
        return x


class VGGNet(fluid.dygraph.Layer):
    """
    VGG网络
    """

    def __init__(self, layers=16):
        """
        构造函数
        :param name_scope:   命名空间
        :param layers:       具体的层数如VGG-16、VGG-19等
        """
        super(VGGNet, self).__init__()
        self.vgg_spec = {
            11: ([1, 1, 2, 2, 2]),
            13: ([2, 2, 2, 2, 2]),
            16: ([2, 2, 3, 3, 3]),
            19: ([2, 2, 4, 4, 4])
        }
        assert layers in self.vgg_spec.keys(), \
            "supported layers are OrderedDict() but input layer is OrderedDict()".format(self.vgg_spec.keys(), layers)

        nums = self.vgg_spec[layers]
        self.conv1 = ConvBlock(self.full_name(), num_channels=3, num_filters=64, groups=nums[0])
        self.conv2 = ConvBlock(self.full_name(), num_channels=64, num_filters=128, groups=nums[1])
        self.conv3 = ConvBlock(self.full_name(), num_channels=128, num_filters=256, groups=nums[2])
        self.conv4 = ConvBlock(self.full_name(), num_channels=256, num_filters=512, groups=nums[3])
        self.conv5 = ConvBlock(self.full_name(), num_channels=512, num_filters=512, groups=nums[4])

        fc_dim = 4096
        self.fc1 = fluid.dygraph.Linear(input_dim=25088, output_dim=fc_dim, act='relu')
        self.fc2 = fluid.dygraph.Linear(input_dim=fc_dim, output_dim=768, act='relu')

    def forward(self, inputs, label=None):
        """前向计算"""
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = fluid.layers.reshape(out, [-1, 25088])

        out = self.fc1(out)
        out = fluid.layers.dropout(out, dropout_prob=0.5)

        out = self.fc2(out)
        out = fluid.layers.dropout(out, dropout_prob=0.5)

        return out