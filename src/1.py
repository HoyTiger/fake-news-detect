import paddle
paddle.__version__ = '1.8'
import paddle.fluid as P
import paddle.fluid.dygraph as D
from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import ErnieModel, ErnieModelForSequenceClassification
import ernie
import numpy as np

D.guard().__enter__()
model = ErnieModel.from_pretrained('ernie-1.0',  num_labels=2)    # Try to get pretrained model from server, make sure you have network connection
model.eval()
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

ids, _ = tokenizer.encode('hello world')
ids = D.to_variable(np.expand_dims(ids, 0))  # insert extra `batch` dimension
pooled, encoded = model(ids)                 # eager execution
len(ids.shape)


params_info = paddle.summary(model, (1,4), 'int64')
print(params_info)