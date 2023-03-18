import math

class Variable:
  def __init__(self, value, _childs=(), _operation = '',label = '' ):
    self.grad = 0.0
    self._backward = lambda: None
    self.label = label
    self.value = value
    self.prev = set(_childs)
    self.operation = _operation

  
  def __repr__(self):
    return f"Value(data = {self.value})"

  def __add__(self,other):
    if (type(other) != Variable): other = Variable(other)
    output = Variable(self.value + other.value)
    t = (self,other)
    output.prev = set (t)
    output.operation = '+'

    def _backward():
      self.grad += 1.0 * output.grad
      other.grad += 1.0 * output.grad
    output._backward = _backward

    return output
  
  def __mul__(self,other):
    if (type(other) != Variable): other = Variable(other)
    output = Variable(self.value * other.value )
    t = (self,other)
    output.prev = set (t)
    output.operation = '*'
    
    def _backward():
      self.grad += other.value * output.grad
      other.grad += self.value * output.grad
    output._backward = _backward

    return output

  def __sub__(self, other): # self - other
    if (type(other) != Variable): other = Variable(other)
    output = Variable(self.value - other.value)
    t = (self,other)
    output.prev = set (t)
    output.operation = '-'

    def _backward():
      self.grad += 1.0 * output.grad
      other.grad += -1.0 * output.grad
    output._backward = _backward
    
    return output

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Variable(self.value**other, (self,), f'**{other}')

    def _backward():
        self.grad += other * (self.value ** (other - 1)) * out.grad
    out._backward = _backward

    return out

  def tanh(self):
    x = self.value
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Variable(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out
  
  def backward(self):
    
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v.prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()