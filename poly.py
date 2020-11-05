import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training set
X_train = [[6], [8], [10], [14], [18]] #diamters of pizzas
y_train = [[7], [9], [13], [17.5], [18]] #prices of pizzas
# Testing set
X_test = [[6], [8], [11], [16]] #diamters of pizzas
y_test = [[8], [12], [15], [18]] #prices of pizzas

fig,ax=plt.subplots(1,2,figsize=(20, 8)) # Creating subplots
regressor = LinearRegression( ) # Train the Linear Regression model and plot a prediction
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
ax[0].plot(xx, yy)

## Trying to find relations to create Polynomial degrees
## Creating train and test accuracy lists  
train_acc=[]
test_acc=[]
for degree in range(2,12,2):
        poly_featurizer = PolynomialFeatures(degree=degree) # Set the degree of the Polynomial Regression model
        X_train_poly = poly_featurizer.fit_transform(X_train)# This transforms input data matrix into a new data matrix 
        X_test_poly = poly_featurizer.transform(X_test)
        regressor_poly = LinearRegression() # Train and test the regressor quadratic model
        regressor_poly=regressor_poly.fit(X_train_poly, y_train)
        xx_poly = poly_featurizer.transform(xx.reshape(xx.shape[0], 1))
        train_score=regressor_poly.score(X_train_poly,y_train) # Calculating accuracy for training and testing data and storing in list
        train_acc.append(train_score)
        test_score=regressor_poly.score(X_test_poly,y_test)
        test_acc.append(test_score)
        ax[0].plot(xx, regressor_poly.predict(xx_poly), linestyle='--',label='Model on degree: {}'.format(degree)) # Plot the graph
# Labelling
        ax[0].legend()
        ax[0].set_title('Pizza price regressed on diameter')
        ax[0].set_xlabel('Diameter in inches')
        ax[0].set_ylabel('Price in dollars')
        ax[0].axis([0, 25, 0, 25])
        ax[0].grid(True)
        ax[0].scatter(X_train, y_train) # Scatter plot for plotting training points

## Plotting for test and train accuracy

ax[1].plot(range(2,12,2),train_acc,label='Training Accuracy')
ax[1].plot(range(2,12,2),test_acc,label='Training Accuracy')
ax[1].set_xlabel('Polynomial Degree')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Training and testing set accuracy w.r.t polynomial model degrees')
ax[1].legend()
ax[1].grid(True)
plt.show()