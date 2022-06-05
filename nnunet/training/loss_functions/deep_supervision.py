#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn
import torch

class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l


if __name__ == '__main__':
    # for testing
    import torch.nn as nn
    import torch
    from torch.autograd import Variable
    from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
    weights = [0.53333333,0.26666667,0.13333333,0.06666667,0.]
    cross_entropy_loss = RobustCrossEntropyLoss(reduction = 'none')
    final_loss = MultipleOutputLoss2(cross_entropy_loss,weights)
    x = [Variable(torch.rand(2,10,64,64)) for _ in range(5)]
    y = [Variable(torch.randint(high = 10,size=(2,1,64,64))) for _ in range(5)]
    a = Variable(torch.tensor(10.), requires_grad=True)
    b = Variable(torch.tensor(3.), requires_grad=True)
    c = Variable(torch.tensor(5.), requires_grad=True)
    prediction = [torch.sum(c*(a * t + b)) for t in x]
    log_prob = final_loss(x,y)
    grads = [torch.autograd.grad(outputs=pred,
                        inputs=(a,b,c), allow_unused=True) for pred in prediction]
    grads= [sum(d**2 for d in p) for p in zip(*grads)]
    print(grads)