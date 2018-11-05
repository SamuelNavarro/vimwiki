# Notes for lesson 7

## Multioutput model:
We use an entire section and use it to predict every one of the following characters. This was to make a better implementation than the previous one because in the previous one we train the model in the same characters a lot of time. 

The next concern is that we throw away the hidden state every time we go to the next section. So let's not throw away the matrix of h.

The problem was:
```python
class CharRNN(nn.Modele):
	.
	.
	.
	def forward(self, *cs):
	.
	h = V(torch.zeros(1, bs, n_hidden))
```

Every time we do a minibatch we begin our hidden states (orange circles).
- This is our new class.
```python 
class CharSeqStatefulRnn(nn.Module):
    def __init__(self, vocab_size, n_fac, bs):
        self.vocab_size = vocab_size
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNN(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)

    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h.size(1) != bs: self.init_hidden(bs)
        outp,h = self.rnn(self.e(cs), self.h)
        self.h = repackage_var(h)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)

    def init_hidden(self, bs): self.h = V(torch.zeros(1, bs, n_hidden))
```

Here we pass in self.init_hidden(bs) to the constructor. 
The line: `self.h = repackage_var(h)`: 
- If we were doing `self.h = h` and now we train it on a doc that is a million characters long. Then the size of the RNN (graph) has a million circles in it. So when we update our weights based on our errors, if we have a million characters, our unroll RNN is a million layers long so we have a 1 million fully conected layer;  so `class CharSeqStatefulRnn` is a 1 million layer fully conected layer. The problem with this is that is going to be very memory intensive. 

### Considerations:
1. So, we want to forget the history from time to time to avoid that. We still are getting the state but we can remember the state without remembering how we got there. That's where use `repackage_var` This approach is back prop through time. 

- `repackage_var`: grab the tensor out of it and create a new variable out of that. so this variable is going to have the same value but no history of operations. We are keeping our hidden state but not our hidden states history.

- Another reason to not backprop through many layers is that we have any kind of gradient instability, we avoid that. A longer value of bptt means your are able to track a longer memory. bptt is something  you get to tune. 

1. How we put our data into this?:

- We want to do a minibatch at a time. We want to look a section a predict the next part of the other one. And at the same time, we want to take an independent section and predict the nex part of other one and so forth. 
- We create 64 equally size minibatches. the minibatches are between the green lines. 
	- Each one is of size bptt. 

![](images/bs_bptt.png)

- Our first minibatch is all marked with a blue circle, and then we predict all of the following (offset by one). 

- *What about data augmentation?* 
  - One approach which won a kaggle competition randomly insert different rows. But this is something that needs to be look into more depth. 
- *How do we choose our bptt?*
 - Your matrix size of minibatch is bpttxminibatchsize. So one problem is that your gpu ram needs to be able to fit that by your embedding matrix. So one thing is you can reduce bptt is you have your loss to NaN because you have less layers. If it's too slow, you can try to decrease your bptt. (There is QRNN which runs this in parallel). 


In `TEXT = data.Filed(lower=True, tokenize=list)` we make sure that each minibatch contains a list of characters. 

`n_hidden`is the size of each of the circles. 
942 batches to go through = Number of tokens / Batch size / bptt.

In practice this is not exactly right. Pytorch randomize the size of bptt a little bit each time. So is going to be slighly different to 8 on avg. 
Is going to be constant per mini batch because we are multiplying h to the size of the minibatc. But the sequence can have different length. (The last minibatch size is probably going to be smaller). We handle this with:
```python
bs = cs[0].size(0)
if self.h.size(1) != bs: self.init_hidden(bs)
```

`TEXT.vocab.itos` is a list of unique characters. TEXT.vocab contains a lot of stuff. 

3. The loss functions such as softmax, are not happy receiving a rank-3 Tensor. We need to flat them out.

![](images/loss_funct_flatten.png)

 - `F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)` That's for the predictions. For the target, torch text knows that the targets needs to be of that shape so it do it for us. 

**Recap**
1. Get rid off the history. 
2. Recreate the hidden state if the batch size changes. 
3. Flatt the predictions out. 
4. Use torch text to create mini batches that line up nicely. 



**GRU**
RNNCell is not use in practice. The reason is that even tanh is used, you tend to find gradient explosions so we have to use pretty low learning rates and pretty small values of bptt. 

So we can replace de RNNCell with a GRUCell 

$$ h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t $$

This equation is in [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

Another usefull [link](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)


