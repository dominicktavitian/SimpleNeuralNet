#
# INFO 445 Script for puppy/kitten preditiction.

#import packages
%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt


#points are in order of width, height, and type (0, 1) 0 is a kitten and dog is a 1 
data = [[2.3,   1.12,   0],
        [3.1,   1.63, 1],
        [4.1,   1.58, 1],
        [3.3,   1.03,  0],
        [2.3,   .52,  0],
        [.9,    1.44,  0],
        [5.9,  1.01,  1],
        [3.8,   .58,  1]]
#predict new pet
new_pet = [2.5, 1]


#GRAPH 1 - make the scatter plot for a visual. 
def scatter_p():
    plt.grid()
    for i in range(len(data)):
        c = 'b'
        if data[i][2] == 0:
            c = 'r'
        plt.scatter([data[i][0]], [data[i][1]], c=c)

    plt.scatter([new_pet[0]], [new_pet[1]], c='green')
scatter_p()


#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
#sigmoid prime function -- derivative = prime
def sigmoid_prime(x): 
    return sigmoid(x) * (1-sigmoid(x))


#GRAPH 2 - plot sigmoids
X = np.linspace(-10, 10, 100)
#sigmoid function plotted for visual in blue
plt.plot(X, sigmoid(X), c="b")
# sigmoid prime function plotted for visual in red. 
fig = plt.plot(X, sigmoid_prime(X), c="r")


# training function! 
def train():
    #randomize the weights and biases
    w1 = np.random.randn()
    w2 = np.random.randn()
    b = np.random.randn()
    
    #iterations and learning rate
    iterations = 1000
    learning_rate = 0.1
    
    #store costs in this list
    costs = []
    
    for i in range(iterations):
        #generate random point to test against
        ri = np.random.randint(len(data))
        point = data[ri]
        
        #z function
        z = point[0] * w1 + point[1] * w2 + b
        prediction = sigmoid(z)
        
        target = point[2]
        
        #cost function
        cost = np.square(prediction - target)
        
        #for loop to see how the cost error goes down. 
        if i % 100 == 0:
            c = 0
            for j in range(len(data)):
                p = data[j]
                p_prediction = sigmoid(w1 * p[0] + w2 * p[1] + b)
                c += np.square(p_prediction - p[2])
            costs.append(c)
        
        #derivatives for adjustment of neural network. 
        der_cost_prediction = 2 * (prediction - target)
        der_prediction_z = sigmoid_prime(z)
        
        der_z_w1 = point[0]
        der_z_w2 = point[1]
        der_z_b = 1
        
        der_cost_z = der_cost_prediction * der_prediction_z
        
        der_cost_w1 = der_cost_z * der_z_w1
        der_cost_w2 = der_cost_z * der_z_w2
        der_cost_b = der_cost_z * der_z_b
        
        w1 = w1 - learning_rate * der_cost_w1
        w2 = w2 - learning_rate * der_cost_w2
        b = b - learning_rate * der_cost_b
        
    return costs, w1, w2, b
        
costs, w1, w2, b = train()

#GRAPH 3 - plot graph of costs
costs_graph = plt.plot(costs)


# predict what the new_pet is!
z = w1 * new_pet[0] + w2 * new_pet[1] + b
prediction = sigmoid(z)

print(prediction)
print("Kitten if close to 0 -- Puppy if close to 1")




