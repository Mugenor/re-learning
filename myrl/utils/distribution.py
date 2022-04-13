import torch as T
import torch.nn.functional as F
from torch import distributions


# ref: https://github.com/kengz/SLM-Lab/blob/faca82c00c51a993e1773e115d5528ffb7ad4ade/slm_lab/lib/distribution.py#L27
class GumbelSoftmax(distributions.RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=T.Size()):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = T.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - T.log(-T.log(u))
        return T.argmax(noisy_logits, dim=-1)

    def rsample(self, sample_shape=T.Size()):
        '''
        Gumbel-softmax resampling using the Straight-Through trick.
        Credit to Ian Temple for bringing this to our attention. To see standalone code of how this works, refer to https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
        '''
        rout = super().rsample(sample_shape)  # differentiable
        out = F.one_hot(T.argmax(rout, dim=-1), self.logits.shape[-1]).float()
        return (out - rout).detach() + rout

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return -T.sum(-value * F.log_softmax(self.logits, -1), -1)