import torch
from torch.autograd import Function
import torch.nn as nn

def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)

def _threshold_and_support(input, dim=0):
    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size

class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, input, dim=0):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

sparsemax = SparsemaxFunction.apply

class Sparsemax(nn.Module):
    def __init__(self, dim=0):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        return sparsemax(input, self.dim)

class LogSparsemax(nn.Module):
    def __init__(self, dim=0):
        super(LogSparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        return torch.log(sparsemax(input, self.dim))
