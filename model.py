import paddle.fluid as F
import paddle.nn as nn
import numpy as np
import paddle
import paddle.fluid as P
import paddle.fluid.dygraph as D
from vgg import *

class ReverseLayerF(D.Layer):
    def __init__(self):
        super(ReverseLayerF, self).__init__()

    def forward(self, x):
        self.lambd = 1
        return x

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x):
    return ReverseLayerF()(x)


class CNN_Fusion(D.Layer):
    def __init__(self):
        super(CNN_Fusion, self).__init__()
        self.init = nn.initializer.TruncatedNormal(0.02)
        self.event_num = 15
        self.vocab_size = 180000
        self.emb_dim = 32
        C = 2
        self.hidden_size = 32
        self.lstm_size = 32
        self.social_size = 19

        # TEXT RNN
        self.embed = nn.Embedding(self.vocab_size,  self.emb_dim)
        self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        self.text_fc = nn.Linear(self.lstm_size, self.hidden_size, weight_attr=paddle.ParamAttr(initializer=self.init))
        self.text_encoder = nn.Linear(self.emb_dim, self.hidden_size, weight_attr=paddle.ParamAttr(initializer=self.init))

        ### TEXT CNN
        channel_in = 1
        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.convs = nn.LayerList([nn.Conv2D(channel_in, filter_num, (K, self.emb_dim)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size, weight_attr=paddle.ParamAttr(initializer=self.init))

        self.dropout = nn.Dropout(0.5)

        # IMAGE
        # hidden_size = args.hidden_dim
        vgg_19 = VGG19(1000)
        params = paddle.load('model3')
        vgg_19.set_dict(params, use_structured_name=True)
        for i in vgg_19.parameters():
            i.stop_gradient = True  
        # visual model
        num_ftrs = 1000
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        # self.image_fc2 = nn.Linear(512, self.hidden_size)
        self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size), weight_attr=paddle.ParamAttr(initializer=self.init))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size, weight_attr=paddle.ParamAttr(initializer=self.init))

        ##ATTENTION
        self.attention_layer = nn.Linear(self.hidden_size, self.emb_dim, weight_attr=paddle.ParamAttr(initializer=self.init))

        ## Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_sublayer('c_fc1', nn.Linear(self.hidden_size*2, 2, weight_attr=paddle.ParamAttr(initializer=self.init)))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        # self.class_classifier.add_module('c_fc2', nn.Linear(self.hidden_size, 2))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(self.hidden_size))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_sublayer('c_softmax', nn.Softmax())

        ###Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_sublayer('d_fc1', nn.Linear(self.hidden_size*2, self.hidden_size, weight_attr=paddle.ParamAttr(initializer=self.init)))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.domain_classifier.add_sublayer('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_sublayer('d_fc2', nn.Linear(self.hidden_size, self.event_num, weight_attr=paddle.ParamAttr(initializer=self.init)))
        self.domain_classifier.add_sublayer('d_softmax', nn.Softmax())

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (D.to_variable(paddle.zeros(1, batch_size, self.lstm_size)),
                D.to_variable(paddle.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = paddle.nn.functional.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        # x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = paddle.nn.functional.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, text, image, mask):
        ### IMAGE #####
        image = self.vgg(image)  # [N, 512]
        image = paddle.nn.functional.leaky_relu(self.image_fc1(image))

        ##########CNN##################
        text = self.embed(text)
        text = text * mask.unsqueeze(2).expand_as(text)
        text = text.unsqueeze(1)
        text = [paddle.nn.functional.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]  # [(N,hidden_dim,W), ...]*len(window_size)
        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,hidden_dim), ...]*len(window_size)
        text = [paddle.nn.functional.max_pool1d(i, i.shape[2]).squeeze(2) for i in text]
        text = paddle.concat(text, 1)
        text = paddle.nn.functional.leaky_relu(self.fc1(text))
        text_image = paddle.concat((text,image), 1)

        ### Fake or real
        class_output = self.class_classifier(text_image)
        ## Domain (which Event )
        reverse_feature = grad_reverse(text_image)
        domain_output = self.domain_classifier(reverse_feature)

        # ### Multimodal
        # text_reverse_feature = grad_reverse(text)
        # image_reverse_feature = grad_reverse(image)
        # text_output = self.modal_classifier(text_reverse_feature)
        # image_output = self.modal_classifier(image_reverse_feature
        return class_output, domain_output

import paddle.fluid as F
import paddle.nn as nn
from src.model import *
import numpy as np
import paddle
import paddle.fluid as P
import paddle.fluid.dygraph as D
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import *
import ernie

class AttentionLayer(nn.Layer):
    def __init__(self, d_model,  name=None):
        super(AttentionLayer, self).__init__()
        init = nn.initializer.TruncatedNormal(0.02)
        self.d_model = d_model
        d_model_q = d_model
        d_model_v = d_model

        self.d_key = d_model_q
        self.q = nn.Linear(d_model, d_model_q,weight_attr=paddle.ParamAttr(initializer=init))
        self.k = nn.Linear(d_model, d_model_q,weight_attr=paddle.ParamAttr(initializer=init))
        self.v = nn.Linear(d_model, d_model_q,weight_attr=paddle.ParamAttr(initializer=init))
        self.o = nn.Linear(d_model_v, d_model,weight_attr=paddle.ParamAttr(initializer=init))

        self.dropout = nn.Dropout(0.5)

    def forward(self, queries, keys, values):
        assert len(queries.shape) == len(keys.shape) == len(values.shape)
        #bsz, q_len, q_dim = queries.shape
        #bsz, k_len, k_dim = keys.shape
        #bsz, v_len, v_dim = values.shape
        #assert k_len == v_len
        
        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)

        
        q = q.scale(self.d_key**-0.5)
        score = q.matmul(k, transpose_y=True)
        score = nn.functional.softmax(score)
        score = self.dropout(score)

        out = score.matmul(v)
        return out


class GRU_Attention(nn.Layer):
    def __init__(self, d_vocab, d_emb, init):
        super(GRU_Attention, self).__init__()
        self.d_emb = d_emb
        self.word_emb = nn.Embedding(
            d_vocab,
            d_emb,
            weight_attr=P.ParamAttr(initializer=init))
        self.text_gru = nn.GRU(d_emb, d_emb, num_layers=2, direction='bidirectional')
        self.attention_linear = AttentionLayer(d_emb, 12)
        self.pool = nn.Linear(
            d_emb,
            d_emb,
            weight_attr=P.ParamAttr(initializer=init))
    
    def forward(self, text):
        embedded = self.word_emb(text)
        embedded = nn.LayerNorm(normalized_shape=self.d_emb, weight_attr=P.ParamAttr(initializer=nn.initializer.Constant(1.)),bias_attr=P.ParamAttr(initializer=nn.initializer.Constant(0.)))(embedded)
        embedded = nn.Dropout(0.2)(embedded)
        gru, _ = self.text_gru(embedded)
        att = attention_linear(gru, gru, gru)
        pooled = F.tanh(self.pooler(att[:, 0, :]))

        return pooled


class BPNet(D.Layer):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2,\
                 n_hidden_3, out_dim):
        super(BPNet, self).__init__()
        self.layer1 = nn.Sequential(D.Linear(in_dim, n_hidden_1))
        self.layer2 = nn.Sequential(D.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm(n_hidden_2), nn.LeakyReLU(), nn.Dropout(0.5))
        self.layer3 = nn.Sequential(D.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm(n_hidden_3), nn.LeakyReLU(), nn.Dropout(0.5))
        self.layer6 = nn.Sequential(D.Linear(n_hidden_3, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer6(x)
        return x

class FakeNewsNet(D.Layer):
    def __init__(self, cin = 768+768+768+768+128):
        super(FakeNewsNet, self).__init__()
        self.cin = cin
        self.ernie_model = ErnieModel.from_pretrained('ernie-1.0',  num_labels=2)
        # self.ernie_model_cls = ErnieModelForSequenceClassification.from_pretrained('ernie-1.0', num_labels=2)
        # self.img_gru = nn.GRU(768, 300, num_layers=2, direction='bidirectional')
        self.vgg = VGGNet(layers=19)
        self.bp_net1 = BPNet(80, 100,100,100, 128)
        self.init =  nn.initializer.TruncatedNormal(0.02)
        self.classify = D.Linear(cin, 2, dtype='float32', param_attr=paddle.ParamAttr(initializer=self.init))
        # self.hidden = D.Linear(cin, cin, dtype='float32', param_attr=paddle.ParamAttr(initializer=self.init),)

        self.attention_linear1 = AttentionLayer(cin)
        # self.ln1 = nn.LayerNorm(normalized_shape=cin, weight_attr=P.ParamAttr(initializer=nn.initializer.Constant(1.)),bias_attr=P.ParamAttr(initializer=nn.initializer.Constant(0.)))
        # self.ln2 = nn.LayerNorm(normalized_shape=cin, weight_attr=P.ParamAttr(initializer=nn.initializer.Constant(1.)),bias_attr=P.ParamAttr(initializer=nn.initializer.Constant(0.)))
        # self.ffn = nn.Sequential(
        #     D.Linear(cin, 4*cin, dtype='float32', param_attr=paddle.ParamAttr(initializer=self.init)),
        #     nn.Softmax(),
        #     nn.Dropout(0.2),
        #     D.Linear(4*cin, cin, dtype='float32', param_attr=paddle.ParamAttr(initializer=self.init))
        # )



    def forward(self, features, text_ids, pic_ids, image):
        
        # return self.ernie_model_cls(text_ids)
        text_out, _ = self.ernie_model(text_ids)
        # text_out = paddle.nn.functional.leaky_relu(text_out)
        text_out = self.ernie_model.dropout(text_out)

        pic_ids_out, _ = self.ernie_model(pic_ids)
        # pic_ids_out = paddle.nn.functional.leaky_relu(pic_ids_out)
        pic_ids_out = self.ernie_model.dropout(pic_ids_out)


        stat_out = self.bp_net1(features)
        # stat_out = paddle.nn.functional.leaky_relu(stat_out)
        stat_out = nn.Dropout(0.2)(stat_out)
        
        img_out = self.vgg(image)
        # img_out = paddle.nn.functional.softmax(img_out)
        img_out = nn.Dropout(0.5)(img_out)
        
        sim = paddle.abs(text_out - pic_ids_out)
        sim = nn.Dropout(0.5)(sim)
        out = paddle.concat(x=[sim, text_out, pic_ids_out, img_out, stat_out], axis=-1)
        # out = self.ln1(out)
        # att_out =  self.attention_linear1(out, out, out)
        # att_out = nn.Dropout(0.2)(att_out)
        # out = att_out + out
        # out = self.ln1(out)
        # ffn_out = self.ffn(out)
        # ffn_out = nn.Dropout(0.2)(ffn_out) 
        # out = ffn_out + out
        # out = self.ln2(out)
        # out = self.text_gru(text_ids)
        out = nn.BatchNorm(768+768+768+768+128)(out)
        # out =  self.attention_linear1(out, out, out)
        # out = nn.leaky_relu(out)
        out = nn.Dropout(0.2)(out)
        out = self.classify(out)
        return out
