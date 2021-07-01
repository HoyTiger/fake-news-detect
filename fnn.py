import sys
sys.path.append('external-libraries')
import argparse
import os
from src.data import *
import numpy as np
import pandas as pd
import jieba
import paddle.fluid.dygraph as D
from sklearn import metrics
from visualdl import LogWriter
from ernie.tokenizing_ernie import ErnieTokenizer

from model import *
import time
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

def encode(text):
    mask_seq = np.zeros(200)
    text_id, _ = tokenizer.encode(text) # ErnieTokenizer 会自动添加ERNIE所需要的特殊token，如[CLS], [SEP]
    text_id = text_id[:200]
    mask_seq[:len(text_id)] = 1.0
    text_id = np.pad(text_id, [0, 200-len(text_id)], mode='constant') # 对所有句子都补长至300，这样会比较费显存；
    return text_id, mask_seq

BATCH=16

from src.data import *

tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

def encode(text):
    text_id, _ = tokenizer.encode(text) # ErnieTokenizer 会自动添加ERNIE所需要的特殊token，如[CLS], [SEP]
    text_id = text_id[:200]
    text_id = np.pad(text_id, [0, 200-len(text_id)], mode='constant') # 对所有句子都补长至300，这样会比较费显存；
    return text_id

def make_data():
    data = []
    text_ids = []
    pic_text_ids = []
    for text in sortedTrain['clean_text']:
        text_id = encode(text)
        text_id = np.array(text_id)
        text_ids.append(text_id)
    for text in sortedTrain['pic_text']:
        text_id = encode(text)
        text_id = np.array(text_id)
        pic_text_ids.append(text_id)


    text_ids = np.array(text_ids).astype('int64')
    feas = np.array(train_use_data).astype('float32')
    imgs = custom_image_reader(sortedTrain['piclist'])
    # imgs = [0] * len(sortedTrain['clean_text'])
    labels = np.array(train_label).astype('float32')
    for fea, text_id,pic_ids, img, label in zip(feas, text_ids, pic_text_ids,imgs,labels):
        # if label == 0:
        #     flag = np.random.randint(3)
        #     if flag == 1:
        #         continue
        fea_id = np.array(fea)
        img = np.array(img)
        text_id = np.array(text_id)
        pic_ids = np.array(pic_ids)
        label_id = np.array(label)
        data.append((fea_id, text_id, pic_ids, img, label_id))
    return np.array(data)


def get_batch_data(data, i):
    d = data[i*BATCH: (i + 1) * BATCH]
    feature, text_id,pic_ids, img, label = zip(*d)
    feature = np.stack(feature)  # 将BATCH行样本整合在一个numpy.array中
    text_id = np.stack(text_id)
    
    pic_ids = np.stack(pic_ids)
    img = np.stack(img)
    label = np.stack(list(label))
    feature = D.to_variable(feature).astype('float32') # 使用to_variable将numpy.array转换为paddle tensor
    text_id = D.to_variable(text_id).astype('int64') # 使用to_variable将numpy.array转换为paddle tensor
    pic_ids = D.to_variable(pic_ids).astype('int64')
    img = D.to_variable(img).astype('float32')
    label = D.to_variable(label).astype('int64')
    return feature,text_id, pic_ids, img, label


place=paddle.fluid.CPUPlace()
D.guard(place=place).__enter__() # 为了让Paddle进入动态图模式，需要添加这一行在最前面
with paddle.fluid.dygraph.guard():
    sortedTrain = pd.read_csv('use_train2.csv')
    sortedTrain = sortedTrain[sortedTrain['piclist'] != '-1']
    train_data =sortedTrain[['emoji_num', 'exclamatory_mark', 'exifinfo_count', 'h', 'l', 'hash_tag_len', 'hash_tag_num',
       'dct_0', 'dct_1', 'dct_2', 'dct_3', 'dct_4', 
       'dct_5', 'dct_6', 'dct_7', 'dct_8', 'dct_9', 'dct_10', 'dct_11',
       'dct_12', 'dct_13', 'dct_14', 'dct_15', 'dct_16', 'dct_17', 'dct_18',
       'dct_19', 'dct_20', 'dct_21', 'dct_22', 'dct_23', 'dct_24', 'dct_25',
       'dct_26', 'dct_27', 'dct_28', 'dct_29', 'dct_30', 'dct_31', 'dct_32',
       'dct_33', 'dct_34', 'dct_35', 'dct_36', 'dct_37', 'dct_38', 'dct_39',
       'dct_40', 'dct_41', 'dct_42', 'dct_43', 'dct_44', 'dct_45', 'dct_46',
       'dct_47', 'dct_48', 'dct_49', 'dct_50', 'dct_51', 'dct_52', 'dct_53',
       'dct_54', 'dct_55', 'dct_56', 'dct_57', 'dct_58', 'dct_59', 'dct_60',
       'dct_61', 'dct_62', 'dct_63', 'neg_prob', 'num_url', 'pic_num', 'pos_prob',
       'question_mark', 'size', 'title_len','userFansCount','userWeiboCount']]
    train_use_data = train_data.fillna(-1)
    train_label = sortedTrain['label']
    sortedTrain = sortedTrain.fillna('')

    sortedTrain['pic_text']  = sortedTrain['general'] + sortedTrain['recognieze_text']
    sortedTrain['pic_text']
    all_data = make_data()
    np.random.shuffle(all_data)
    f = open('result.txt','w')
    eval_data = all_data[-(len(all_data))//5:] 
    train_data = all_data[(len(all_data))//5:-(len(all_data))//5] 
    test_data = all_data[:(len(all_data))//5]
    model = FakeNewsNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = paddle.fluid.optimizer.Adam(5e-5,parameter_list=model.parameters())
    best_validate_acc = 0.000
    best_test_acc = 0.000
    best_loss = 100
    best_validate_dir = ''
    best_list = [0, 0]
    early_stop = 0
    with LogWriter(logdir="logs/log-fnn") as writer:
        print('training model')
        adversarial = True
        cnt = 0
        cnt2 = 0
        for epoch in range(100):
            np.random.shuffle(train_data)
            p = float(epoch) / 100
            # lambd = 2. / (1. + np.exp(-10. * p)) - 1
            # lr = 0.001 / (1. + 10 * p) ** 0.75

            # optimizer.lr = lr
            # rgs.lambd = lambd
            start_time = time.time()
            cost_vector = []
            class_cost_vector = []
            domain_cost_vector = []
            acc_vector = []
            valid_acc_vector = []
            test_acc_vector = []
            vali_cost_vector = []
            test_cost_vector = []
            for j in range(len(train_data) // BATCH):
                model.train()
                feature, text_id, pic_ids, img, train_labels = get_batch_data(train_data, j)
                if len(train_labels.shape) == 1:
                        train_labels = paddle.fluid.layers.reshape(train_labels, [-1, 1])

                # Forward + Backward + Optimize
                class_outputs = model(feature, text_id, pic_ids, img)

                ## Fake or Real loss
                class_loss = criterion(class_outputs, train_labels)
                loss = class_loss
                loss = paddle.fluid.layers.reduce_mean(loss)
                loss.backward()
                optimizer.minimize(loss)
                model.clear_gradients()
                argmax = paddle.fluid.layers.argmax(class_outputs, -1)


                accuracy = metrics.accuracy_score(train_labels.numpy().astype('int64'), argmax.numpy().astype('int64'))


                class_cost_vector.append(class_loss.numpy())
            
                cost_vector.append(loss.numpy())
                acc_vector.append(accuracy)
                writer.add_scalar(tag="train-acc", step=cnt, value=accuracy)
                writer.add_scalar(tag="train-loss", step=cnt, value=loss.numpy())
                cnt+=1
                if j%50==0:
                    print('Epoch [%d/%d],batch: %d  Loss: %.4f, Train_Acc: %.4f,'
                % (
                    epoch + 1, 100, j ,loss.numpy(), accuracy))

            
            with D.base._switch_tracer_mode_guard_(is_train=False):
                model.eval()

                validate_acc_vector_temp = []
                for j in range(len(eval_data) // BATCH):
                    feature, text_id, pic_ids, img, validate_labels = get_batch_data(eval_data, j)
                    if len(validate_labels.shape) == 1:
                            validate_labels = paddle.fluid.layers.reshape(validate_labels, [-1, 1])
                    validate_outputs = model(feature, text_id, pic_ids, img)
                    validate_argmax = paddle.fluid.layers.argmax(validate_outputs, -1)
                    vali_loss = criterion(validate_outputs, validate_labels)
                    
                    validate_accuracy = metrics.accuracy_score(validate_labels.numpy().astype('int64'), validate_argmax.numpy().astype('int64'))

                    vali_cost_vector.append(vali_loss.numpy())
                    validate_acc_vector_temp.append(validate_accuracy)
                    writer.add_scalar(tag="eval-acc", step=cnt2, value=validate_accuracy)
                    writer.add_scalar(tag="eval-loss", step=cnt2, value=vali_loss.numpy())
                    cnt2+=1

                validate_acc = np.mean(validate_acc_vector_temp)
                valid_acc_vector.append(validate_acc)
                model.train()
                print('Epoch [%d/%d],  Loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
                    % (
                        epoch + 1, 100, np.mean(cost_vector),
                        np.mean(acc_vector), validate_acc))

                if validate_acc > best_validate_acc:
                    early_stop = 0
                    best_validate_acc = validate_acc
                    if not os.path.exists('model/'):
                        os.mkdir('model/')

                    best_validate_dir = 'model/fnn/bset'
                    D.save_dygraph(model.state_dict(), best_validate_dir)
                else:
                    early_stop += 1
                    if early_stop == 10:
                        print(f'early_stop')
                        break

                duration = time.time() - start_time
                print ('Epoch: %d, Mean_Cost: %.4f, Duration: %.4f, Mean_Train_Acc: %.4f, Mean_vail_Acc: %.4f'
                % (epoch + 1, np.mean(cost_vector), duration, np.mean(acc_vector), np.mean(valid_acc_vector)))
        with D.base._switch_tracer_mode_guard_(is_train=False):
            # Test the Model
            print('testing model')
            model =  FakeNewsNet()
            model_dict, _ = paddle.fluid.load_dygraph(best_validate_dir)
            model.load_dict(model_dict)
            #    print(torch.cuda.is_available())
            model.eval()
            test_score = []
            test_pred = []
            test_true = []
            for j in range(len(test_data) // BATCH):
                feature, text_id, pic_ids, img, test_labels = get_batch_data(test_data, j)
                if len(test_labels.shape) == 1:
                    test_labels = paddle.fluid.layers.reshape(test_labels, [-1, 1])

                test_outputs = model(feature, text_id, pic_ids, img)
                test_argmax = paddle.fluid.layers.argmax(test_outputs, -1)
                if j == 0:
                    test_score = test_outputs.squeeze().numpy()
                    test_pred = test_argmax.squeeze().numpy()
                    test_true = test_labels.squeeze().numpy()
                else:
                    test_score = np.concatenate((test_score, test_outputs.squeeze().numpy()), axis=0)
                    test_pred = np.concatenate((test_pred, test_argmax.squeeze().numpy()), axis=0)
                    test_true = np.concatenate((test_true, test_labels.squeeze().numpy()), axis=0)

            test_accuracy = metrics.accuracy_score(test_true, test_pred)
            test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
            test_precision = metrics.precision_score(test_true, test_pred, average='macro')
            test_recall = metrics.recall_score(test_true, test_pred, average='macro')
            test_score_convert = [x[1] for x in test_score]
            test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')

            test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

            print("Classification Acc: %.4f, AUC-ROC: %.4f"
                % (test_accuracy, test_aucroc))
            print("Classification report:\n%s\n"
                % (metrics.classification_report(test_true, test_pred,  digits=4)))
            print("Classification confusion matrix:\n%s\n"
                % (test_confusion_matrix))


