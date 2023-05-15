import NN_Creater as nn
import Aux_Functions as aux_func

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

#print (my_nn.parameters())
print ("total parameters",len(my_nn.parameters()))

## Every iteration of this loop is an epoch !!
for i in range(1,35):
  ypred = aux_func.forward_pass(my_nn,xs)

  loss = aux_func.calc_loss(ydesired,ypred)
  print(loss)

  aux_func.back_prop(my_nn,loss)

  aux_func.nudge_the_weightsbiases(my_nn)




