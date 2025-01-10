# SETUP
from torch import tensor, stack, softmax, logsumexp
from torch import logaddexp as lae
def lse(*inputs): return logsumexp(stack(inputs), dim=0)  # logsumexp applied to scalars
inf = tensor(float('inf'))
two = tensor(2.)
def leaf(value): return tensor(value, requires_grad=True)

# FIRST PROBLEM
# x = leaf(float('-inf'))
# lae(lae(x, -inf), two).backward()
# print(x.grad.item())

# x = leaf(float('-inf'))
# lae(lse(x, -inf), two).backward()
# print(x.grad.item())

# x = leaf(float('-inf'))
# lae(x, lae(-inf, two)).backward()
# print(x.grad.item())

x = leaf(float('-inf'))
lae(lse(x, -inf),two).backward()
print(x.grad.item())