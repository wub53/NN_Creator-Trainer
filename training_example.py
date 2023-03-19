import NN_Creater as nn

def flush_gradients(mlp):
  for i in mlp.parameters():
      i.grad = 0

def nudge_the_weightsbiases(mlp):
  for i in mlp.parameters():
      i.value += -0.1 * i.grad

def forward_pass(mlp,inputs):
  ypred = [mlp(x) for x in inputs]
  return ypred

def back_prop(mlp,loss):
  flush_gradients(mlp)
  loss.backward() 

def calc_loss(ydesired, ypred):
  # L is the list of loss for each input example
  L = [(yout - ygt)**2 for ygt,yout in zip(ydesired, ypred)]
  #Summing the loss for every 
  loss = nn.Variable(0)
  for i in L:
    loss = loss + i
  return loss
  

# NN or (Multi level perceptron if you may) will be of 3,4,4,1 neurons per layer(4) last one being the output layer
my_nn = nn.MLP(3, [4, 4, 1])

#xs is the list of inputs we are going to feed the my_nn 
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ydesired = [1.0, -1.0, -1.0, 1.0] # desired targets

ypred=[]

print (my_nn.parameters())
print (len(my_nn.parameters()))

## Every iteraton of this loop is an epoch !!
# for i in range(1,15):
#   ypred = forward_pass(my_nn,xs)

#   loss = calc_loss(ydesired,ypred)
#   print(loss)

#   back_prop(my_nn,loss)

#   nudge_the_weightsbiases(my_nn)




