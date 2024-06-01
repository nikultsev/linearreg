import numpy as np
import matplotlib.pyplot as plt
# print(data[0])
# print(data[1])
# plt.scatter(data[0], data[1], marker = '+')

# plt.show()

class Linearreg:
    def __init__(self, x_data , y_data, alpha):
        self.x_data = x_data
        self.y_data = y_data
        self.alpha = alpha # how big you want the steps to be 
        self.m, self.b = [0, 0] # initialise the line as the x axis
    
    def cost_function(self):
        model = lambda x: self.m*x + self.b # regression model (line)
        return 1/(2 * len(self.x_data)) * np.sum((model(self.x_data) - self.y_data)**2) # first term is prediction with model
    # the squared mean error (mean is first fraction term), second observed


    def do_regression(self, max_iterations=1000, tolerance =0.0001):
        # tolerance is how small you want the cost to be for your final guess (how close to the ideal perfect solution which you
        # are approximating)
        # iterations is maximum number of steps that we want our algorithm to try
        convergence = False
        iterations = 0

        while not convergence and iterations < max_iterations: # while not converged, and we are below max iterations    
            model = lambda x: self.m * x + self.b # restate the model
            predictions = model(self.x_data) # use model on data to create predicted line

            error = predictions - self.y_data # difference for each point
            m_gradient = 1/len(self.x_data) * np.sum(error* self.x_data) # if you take derivative of cost function m and b you
            # get these (sub in the model definition and do it carefully)
            b_gradient = 1/len(self.x_data) * np.sum(error)

            self.m = self.m - self.alpha*m_gradient # adjust the the current m variable - scaled by teh gradient and (gradient descent)
            self.b = self.b - self.alpha*b_gradient

            cost = self.cost_function() # calculate the current cost
            
            print('this iteration', cost, self.m, self.b)

            if cost < tolerance: # if the cost is small enough - we say we have converged
                convergence = True

            iterations = iterations + 1 # if we dont converged we do it again

        return self.m, self.b # after all is done return optimised variables


def main():
    data = np.array([[1,3], [2,7], [0.5, 2.7], [10, 15], [-1, 0.3], [7, 8]])
    # data1 = [[1, 1], [2,2]]
    data = np.transpose(data)
    lin = Linearreg(data[0], data[1], 0.05)
    m, b = Linearreg.do_regression(lin)
    line_x = np.linspace(-10, 20, 50)
    model = lambda x: m*x + b
    
    print('This is final model:', m, b)
    
    plt.scatter(data[0], data[1], marker = 'X', color = 'black')
    plt.plot(line_x, model(line_x))
    plt.show()

if __name__ == "__main__":
    main()





