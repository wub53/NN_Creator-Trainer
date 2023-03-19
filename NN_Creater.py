import random

from NN_engine import Variable

class Neuron:
  
  def __init__(self, connections):
    self.w = [Variable(random.uniform(-1,1)) for iter in range(connections)]
    self.b = Variable(random.uniform(-1,1))
  
  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    squished_act = act.tanh()
    return squished_act
  
  def parameters(self):
    return self.w + [self.b]

class Layer:
  
  def __init__(self, connections, num_neurons):
    self.neurons = [Neuron(connections) for iter in range(num_neurons)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    p = []
    for neuron in self.neurons:
      for params in neuron.parameters():
        p.append(params)
    return p
    # return [p for neuron in self.neurons for p in neuron.parameters()]   ##

class MLP:
  
  def __init__(self, connections, layers_exluding_ip):
    mlp_skeleton = [connections] + layers_exluding_ip
    self.layers = [Layer(mlp_skeleton[i], mlp_skeleton[i+1]) for i in range(len(layers_exluding_ip))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    p = []
    for layer in self.layers:
      for params in layer.parameters():
        p.append(params) 
    return p
    #return [p for layer in self.layers for p in layer.parameters()]