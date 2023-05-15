import NN_Creater as nn

# Remember to flush the gradients of the nodes before doing thge back prop
def flush_gradients(mlp):
  for i in mlp.parameters():
      i.grad = 0

# Nudge the weights and biases in the direction with their gradients 2
def nudge_the_weightsbiases(mlp):
  for i in mlp.parameters():
      i.value += -0.1 * i.grad

# Supply the new inputs to the network 
def forward_pass(mlp,inputs):
  ypred = [mlp(x) for x in inputs]
  return ypred

# Initiate the back propogation to calculate the gradients 
def back_prop(mlp,loss):
  flush_gradients(mlp)
  loss.backward() 

# calculate the loss function value for each pass 
def calc_loss(ydesired, ypred):
  # L is the list of loss for each input example
  L = [(yout - ygt)**2 for ygt,yout in zip(ydesired, ypred)]
  #Summing the loss for every 
  loss = nn.Variable(0)
  for i in L:
    loss = loss + i
  return loss
  