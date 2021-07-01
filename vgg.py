import paddle
from paddle import nn

class VGG19(paddle.nn.Layer):
    def __init__(self, num_classes=1000):
        super(VGG19, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
           nn.Linear(4096, num_classes)
            )
        # self.x2paddle_classifier_0_weight = self.create_parameter(shape=[4096, 25088], attr='x2paddle_classifier_0_weight', dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=0.0))
        # self.x2paddle_classifier_0_bias = self.create_parameter(shape=[4096], attr='x2paddle_classifier_0_bias', dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=0.0))
        # self.x2paddle_classifier_3_weight = self.create_parameter(shape=[4096, 4096], attr='x2paddle_classifier_3_weight', dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=0.0))
        # self.x2paddle_classifier_3_bias = self.create_parameter(shape=[4096], attr='x2paddle_classifier_3_bias', dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=0.0))
        # self.x2paddle_classifier_6_weight = self.create_parameter(shape=[1000, 4096], attr='x2paddle_classifier_6_weight', dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=0.0))
        # self.x2paddle_classifier_6_bias = self.create_parameter(shape=[1000], attr='x2paddle_classifier_6_bias', dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.conv0 = paddle.nn.Conv2D(in_channels=3, out_channels=64, kernel_size=[3, 3], padding=1)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=[3, 3], padding=1)
        self.relu1 = paddle.nn.ReLU()
        self.pool0 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=64, out_channels=128, kernel_size=[3, 3], padding=1)
        self.relu2 = paddle.nn.ReLU()
        self.conv3 = paddle.nn.Conv2D(in_channels=128, out_channels=128, kernel_size=[3, 3], padding=1)
        self.relu3 = paddle.nn.ReLU()
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2)
        self.conv4 = paddle.nn.Conv2D(in_channels=128, out_channels=256, kernel_size=[3, 3], padding=1)
        self.relu4 = paddle.nn.ReLU()
        self.conv5 = paddle.nn.Conv2D(in_channels=256, out_channels=256, kernel_size=[3, 3], padding=1)
        self.relu5 = paddle.nn.ReLU()
        self.conv6 = paddle.nn.Conv2D(in_channels=256, out_channels=256, kernel_size=[3, 3], padding=1)
        self.relu6 = paddle.nn.ReLU()
        self.conv7 = paddle.nn.Conv2D(in_channels=256, out_channels=256, kernel_size=[3, 3], padding=1)
        self.relu7 = paddle.nn.ReLU()
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2)
        self.conv8 = paddle.nn.Conv2D(in_channels=256, out_channels=512, kernel_size=[3, 3], padding=1)
        self.relu8 = paddle.nn.ReLU()
        self.conv9 = paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=[3, 3], padding=1)
        self.relu9 = paddle.nn.ReLU()
        self.conv10 = paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=[3, 3], padding=1)
        self.relu10 = paddle.nn.ReLU()
        self.conv11 = paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=[3, 3], padding=1)
        self.relu11 = paddle.nn.ReLU()
        self.pool3 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2)
        self.conv12 = paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=[3, 3], padding=1)
        self.relu12 = paddle.nn.ReLU()
        self.conv13 = paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=[3, 3], padding=1)
        self.relu13 = paddle.nn.ReLU()
        self.conv14 = paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=[3, 3], padding=1)
        self.relu14 = paddle.nn.ReLU()
        self.conv15 = paddle.nn.Conv2D(in_channels=512, out_channels=512, kernel_size=[3, 3], padding=1)
        self.relu15 = paddle.nn.ReLU()
        self.pool4 = paddle.nn.MaxPool2D(kernel_size=[2, 2], stride=2)
        self.pool5 = paddle.nn.AvgPool2D(kernel_size=[1, 1], stride=1, exclusive=True)
        self.relu16 = paddle.nn.ReLU()
        self.relu17 = paddle.nn.ReLU()

    def forward(self, x0):
        x2paddle_input_1 = paddle.to_tensor(data=x0)

        x2paddle_39 = self.conv0(x2paddle_input_1)
        x2paddle_40 = self.relu0(x2paddle_39)
        x2paddle_41 = self.conv1(x2paddle_40)
        x2paddle_42 = self.relu1(x2paddle_41)
        x2paddle_43 = self.pool0(x2paddle_42)
        x2paddle_44 = self.conv2(x2paddle_43)
        x2paddle_45 = self.relu2(x2paddle_44)
        x2paddle_46 = self.conv3(x2paddle_45)
        x2paddle_47 = self.relu3(x2paddle_46)
        x2paddle_48 = self.pool1(x2paddle_47)
        x2paddle_49 = self.conv4(x2paddle_48)
        x2paddle_50 = self.relu4(x2paddle_49)
        x2paddle_51 = self.conv5(x2paddle_50)
        x2paddle_52 = self.relu5(x2paddle_51)
        x2paddle_53 = self.conv6(x2paddle_52)
        x2paddle_54 = self.relu6(x2paddle_53)
        x2paddle_55 = self.conv7(x2paddle_54)
        x2paddle_56 = self.relu7(x2paddle_55)
        x2paddle_57 = self.pool2(x2paddle_56)
        x2paddle_58 = self.conv8(x2paddle_57)
        x2paddle_59 = self.relu8(x2paddle_58)
        x2paddle_60 = self.conv9(x2paddle_59)
        x2paddle_61 = self.relu9(x2paddle_60)
        x2paddle_62 = self.conv10(x2paddle_61)
        x2paddle_63 = self.relu10(x2paddle_62)
        x2paddle_64 = self.conv11(x2paddle_63)
        x2paddle_65 = self.relu11(x2paddle_64)
        x2paddle_66 = self.pool3(x2paddle_65)
        x2paddle_67 = self.conv12(x2paddle_66)
        x2paddle_68 = self.relu12(x2paddle_67)
        x2paddle_69 = self.conv13(x2paddle_68)
        x2paddle_70 = self.relu13(x2paddle_69)
        x2paddle_71 = self.conv14(x2paddle_70)
        x2paddle_72 = self.relu14(x2paddle_71)
        x2paddle_73 = self.conv15(x2paddle_72)
        x2paddle_74 = self.relu15(x2paddle_73)
        x2paddle_75 = self.pool4(x2paddle_74)
        x2paddle_76 = self.pool5(x2paddle_75)
        x2paddle_77 = paddle.flatten(x2paddle_76, 1)
        out = self.classifier(x2paddle_77)
        return out

# def main(x0):
#     # 共1个输入
#     # x0: 形状为[1, 3, 224, 224]，类型为float32。
#     paddle.disable_static()
#     params = paddle.load('model3')
#     model = ONNXModel(1000)
#     model.set_dict(params, use_structured_name=True)
#     model.eval()
#     out = model(x0)
#     paddle.summary(model, (1, 3, 224, 224))
#     return out

# x = paddle.randn((1, 3, 224, 224))
# main(x)



