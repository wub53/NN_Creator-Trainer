import graphviz
from IPython.display import Image

mlp_skeleton = [3, 6, 6, 6, 1]
layers = len(mlp_skeleton)

mlp_tensor = []
for index, value in enumerate (mlp_skeleton):
    l = ['L' + f'{index}_' + str(n) for n in range(0,value)]
    mlp_tensor.append(l)

#print(mlp_tensor)

# Create a Graphviz graph
graph = graphviz.Digraph()

# Displya the Neural Network Layer to Layer
for i in range(len(mlp_tensor)):
    for j in range(len(mlp_tensor[i])):
        if ( i + 1 == len(mlp_tensor)):
           break
        for k in range (len(mlp_tensor[i+1])):
          if (len(mlp_tensor[i+1]) == 1):
             graph.edge(mlp_tensor[i][j], mlp_tensor[i+1][0])
             continue  
          graph.edge(mlp_tensor[i][j], mlp_tensor[i+1][k])

# Render and save the graph
graph.format = 'png'
graph.render('neural_network')

# Display the saved image
Image(filename='neural_network.png')