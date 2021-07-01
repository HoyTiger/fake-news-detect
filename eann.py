import sys
sys.path.append('external-libraries')
import argparse
import os
from src.data import *
import numpy as np
import pandas as pd
import jieba
from gensim.models import Word2Vec
import paddle.fluid.dygraph as D
from sklearn import metrics
from visualdl import LogWriter
from ernie.tokenizing_ernie import ErnieTokenizer

from model import *
import time


BATCH=32

def w2v():
    # word_list = []
    # for sentence in sortedTrain['clean_text']:
    #     for word in sentence:
    #         word_list.append(word)
    min_count = 1
    size = 32
    window = 4

    w2v = Word2Vec(list(sortedTrain['clean_text']), min_count=min_count, size=size, window=window)
    temp = {}
    for i, word in enumerate(w2v.wv.index2word):
        temp[word] = i
    w2v = temp

    data = []
    mask = []
    for l in sortedTrain['clean_text']:
        line_data = []
        seq_len = 200
        mask_seq = np.zeros(200)
        mask_seq[:len(l)] = 1.0
        for word in l:
            line_data.append(w2v[word])
        line_data = line_data[:200]
        line_data = np.pad(line_data, (0, 200 - len(line_data)), mode='constant')
        data.append(line_data)
        mask.append(mask_seq)
    return np.array(data), np.array(mask)

def make_data():
    data = []
    text_ids, masks = w2v()
    print(text_ids.shape)


    imgs = custom_image_reader(sortedTrain['piclist'])
    # imgs = [0] * len(sortedTrain['clean_text'])

    labels = np.array(sortedTrain['label']).astype('int64')
    event_labels =  np.array(sortedTrain['event_label']).astype('int64')
    for text_id, img, mask, label,  event_label in zip(text_ids, imgs, masks, labels, event_labels):
        img = np.array(img)
        text_id = np.array(text_id)
        label_id = np.array(label)
        event_label = np.array(event_label)
        mask = np.array(mask)
        data.append((text_id, img, mask, label_id, event_label))
    return np.array(data)

def get_batch_data(data, i):
    d = data[i * BATCH: (i + 1) * BATCH]
    text_id, img, mask, label, event_label = zip(*d)
    text_id = np.stack(text_id)
    event_label = np.stack(event_label)
    img = np.stack(img)
    label = np.stack(list(label))
    text_id = D.to_variable(text_id).astype('int64')  # 使用to_variable将numpy.array转换为paddle tensor
    event_label = D.to_variable(event_label).astype('int64')
    mask = D.to_variable(mask).astype('float32')
    img = D.to_variable(img).astype('float32')
    label = D.to_variable(label).astype('int64')
    return  text_id, img, mask, label, event_label

place=paddle.fluid.CPUPlace()

sortedTrain = pd.read_csv('1.csv')
sortedTrain = sortedTrain[sortedTrain['piclist'] != '-1']
all_data = make_data()

f = open('result.txt','w')
from sklearn.model_selection import KFold
kf = KFold(n_splits=5,shuffle=True)
train_eval = all_data[(len(all_data))//5:] 
test_data = all_data[:(len(all_data))//5]
D.guard(place=place).__enter__() # 为了让Paddle进入动态图模式，需要添加这一行在最前面
with paddle.fluid.dygraph.guard():
    for index, (train_index, eval_index) in enumerate(kf.split(range(len(train_eval)))):   
        train_data = train_eval[train_index]
        eval_data = train_eval[eval_index]
        model = CNN_Fusion()
        early_stop=0
        criterion = nn.CrossEntropyLoss()
        optimizer = paddle.fluid.optimizer.Adam(5e-5,parameter_list=model.parameters())
        best_validate_acc = 0.000
        best_test_acc = 0.000
        best_loss = 100
        best_validate_dir = ''
        best_list = [0, 0]
        with LogWriter(logdir="logs/log-eann") as writer:
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
                    train_text, train_image, train_mask, train_labels, event_labels = get_batch_data(train_data, j)
                    if len(train_labels.shape) == 1:
                            train_labels = paddle.fluid.layers.reshape(train_labels, [-1, 1])
                            event_labels = paddle.fluid.layers.reshape(event_labels, [-1, 1])

                    # Forward + Backward + Optimize
                    class_outputs, domain_outputs = model(train_text, train_image, train_mask)

                    ## Fake or Real loss
                    class_loss = criterion(class_outputs, train_labels)
                    # Event Loss
                    domain_loss = criterion(domain_outputs, event_labels)
                    loss = class_loss + domain_loss
                    loss = paddle.fluid.layers.reduce_mean(loss)
                    loss.backward()
                    optimizer.minimize(loss)
                    model.clear_gradients()
                    argmax = paddle.fluid.layers.argmax(class_outputs, -1)

                    cross_entropy = True


                    accuracy = metrics.accuracy_score(train_labels.numpy().astype('int64'), argmax.numpy().astype('int64'))


                    class_cost_vector.append(class_loss.numpy())
                    domain_cost_vector.append(domain_loss.numpy())
                
                    cost_vector.append(loss.numpy())
                    acc_vector.append(accuracy)
                    writer.add_scalar(tag="train-acc", step=cnt, value=accuracy)
                    writer.add_scalar(tag="train-loss", step=cnt, value=loss.numpy())
                    writer.add_scalar(tag="train-envent-loss", step=cnt, value=domain_loss.numpy())
                    writer.add_scalar(tag="train-cls-loss", step=cnt, value=class_loss.numpy())
                    cnt+=1
                    if j%50==0:
                        print('Epoch [%d/%d],batch: %d  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, Train_Acc: %.4f,'
                    % (
                        epoch + 1, 100, j ,loss.numpy(), class_loss.numpy(),
                        domain_loss.numpy(),
                    accuracy))

                    # if i == 0:
                    #     train_score = to_np(class_outputs.squeeze())
                    #     train_pred = to_np(argmax.squeeze())
                    #     train_true = to_np(train_labels.squeeze())
                    # else:
                    #     class_score = np.concatenate((train_score, to_np(class_outputs.squeeze())), axis=0)
                    #     train_pred = np.concatenate((train_pred, to_np(argmax.squeeze())), axis=0)
                    #     train_true = np.concatenate((train_true, to_np(train_labels.squeeze())), axis=0)
                with D.base._switch_tracer_mode_guard_(is_train=False):
                    model.eval()

                    validate_acc_vector_temp = []
                    for j in range(len(eval_data) // BATCH):
                        validate_text, validate_image, validate_mask, validate_labels, event_labels = get_batch_data(eval_data, j)
                        if len(validate_labels.shape) == 1:
                                validate_labels = paddle.fluid.layers.reshape(validate_labels, [-1, 1])
                                event_labels = paddle.fluid.layers.reshape(event_labels, [-1, 1])
                        validate_outputs, domain_outputs = model(validate_text, validate_image, validate_mask)
                        validate_argmax = paddle.fluid.layers.argmax(validate_outputs, -1)
                        vali_loss = criterion(validate_outputs, validate_labels)
                        # domain_loss = criterion(domain_outputs, event_labels)
                        # _, labels = torch.max(validate_labels, 1)
                        validate_accuracy = metrics.accuracy_score(validate_labels.numpy().astype('int64'), validate_argmax.numpy().astype('int64'))

                        vali_cost_vector.append(vali_loss.numpy())
                        # validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
                        validate_acc_vector_temp.append(validate_accuracy)
                        writer.add_scalar(tag="eval-acc", step=cnt2, value=validate_accuracy)
                        writer.add_scalar(tag="eval-loss", step=cnt2, value=vali_loss.numpy())
                        cnt2+=1

                    validate_acc = np.mean(validate_acc_vector_temp)
                    valid_acc_vector.append(validate_acc)
                    model.train()
                    print('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
                        % (
                            epoch + 1, 100, np.mean(cost_vector), np.mean(class_cost_vector),
                            np.mean(domain_cost_vector),
                            np.mean(acc_vector), validate_acc))

                    
                    if validate_acc > best_validate_acc:
                        early_stop = 0
                        best_validate_acc = validate_acc
                        if not os.path.exists('model/'):
                            os.mkdir('model/')

                        best_validate_dir = 'model/eann/' +str(index)
                        D.save_dygraph(model.state_dict(), best_validate_dir)
                    else:
                        early_stop += 1
                        if early_stop == 5:
                            print(f'early_stop')
                            break

                    duration = time.time() - start_time
                    print ('Epoch: %d, Mean_Cost: %.4f, Duration: %.4f, Mean_Train_Acc: %.4f, Mean_vail_Acc: %.4f'
                    % (epoch + 1, np.mean(cost_vector), duration, np.mean(acc_vector), np.mean(valid_acc_vector)))
            with D.base._switch_tracer_mode_guard_(is_train=False):
                # Test the Model
                print('testing model')
                model = CNN_Fusion()
                model_dict, _ = paddle.fluid.load_dygraph(best_validate_dir)
                model.load_dict(model_dict)
                #    print(torch.cuda.is_available())
                model.eval()
                test_score = []
                test_pred = []
                test_true = []
                for j in range(len(test_data) // BATCH):
                    test_text, test_image, test_mask, test_labels, event_labels = get_batch_data(test_data, j)
                    if len(test_labels.shape) == 1:
                        test_labels = paddle.fluid.layers.reshape(test_labels, [-1, 1])
                        event_labels = paddle.fluid.layers.reshape(event_labels, [-1, 1])
                    test_outputs, domain_outputs = model(test_text, test_image, test_mask)
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
                    % (metrics.classification_report(test_true, test_pred)))
                f.write("Classification report:\n%s\n"
                    % (metrics.classification_report(test_true, test_pred)))
                
                print("Classification confusion matrix:\n%s\n"
                    % (test_confusion_matrix))


