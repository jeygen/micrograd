from micrograd.engine import *
from micrograd.nn import *

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
#print(n(x))
n(x)

xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0] # desired targets
# forward pass
ypred = [n(x) for x in xs]
print(ypred)
"""
[Value(data=-1.0561083896291652, grad=0), Value(data=-0.36282059113214016, grad=0),
Value(data=-0.14998541424601153, grad=0), Value(data=-0.3537109111330247, grad=0)]

want these to hit targets, how do we do that?


the loss is a single value that rates the perfomance of the neural net
"""

# one way to determine loss
# output - 'ground truth/target'
print(f"loss")
loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
print(loss)

# want to get loss to 0

# take partial derivs
# back pass
loss.backward()

# can access data and weight of any given neuron
n.layers[0].neurons[0].w[0].data

# nudge all grad to lesser slope
for p in n.parameters():
    p.data += -0.01 * p.grad # 0.01 is the 'learning rate'

ypred = [n(x) for x in xs]
print(ypred)
print(f"loss")
loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
print(loss) # this loss should be slightly closer to 0

# forward pass, backward pass, update. this is gradient descent

# when loss gets very low the predictions will be very accurate

# this would be done in a loop

# make sure all grads are reset to 0 each pass, so grad values don't accumulate
