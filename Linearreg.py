import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1,3], [2,7], [0.5, 2.7], [10, 15], [-1, 0.3], [7, 8]])
# data1 = [[1, 1], [2,2]]
data = np.transpose(data)
# print(data[0])
# print(data[1])


# plt.scatter(data[0], data[1], marker = '+')

# plt.show()



class Linearreg:
    def __init__(self, x_data , y_data, alpha):
        self.x_data = x_data
        self.y_data = y_data
        self.alpha = alpha
        self.m, self.b = [0, 0]
    
    def cost_function(self):
        model = lambda x: self.m*x + self.b
        return 1/(2 * len(self.x_data)) * np.sum((model(self.x_data) - self.y_data)**2)


    def do_regression(self, max_iterations=1000, tolerance =0.0001):
        
        convergence = False
        iterations = 0

        while not convergence and iterations < max_iterations:    
            model = lambda x: self.m * x + self.b
            predictions = model(self.x_data)

            error = predictions - self.y_data
            m_gradient = 1/len(self.x_data) * np.sum(error* self.x_data)
            b_gradient = 1/len(self.x_data) * np.sum(error)

            self.m = self.m - self.alpha*m_gradient
            self.b = self.b - self.alpha*b_gradient

            cost = self.cost_function()
            
            print('this iteration', cost, self.m, self.b)

            if cost < tolerance:
                convergence = True

            iterations = iterations + 1

        return self.m, self.b

            
                

        
        
    

lin = Linearreg(data[0], data[1], 0.05)
m, b = Linearreg.do_regression(lin)

line_x = np.linspace(-10, 20, 50)
model = lambda x: m*x + b
plt.scatter(data[0], data[1], marker = 'X', color = 'black')
plt.plot(line_x, model(line_x))

plt.show()





