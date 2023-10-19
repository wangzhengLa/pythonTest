import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

data=pd.read_csv('../data/world-happiness-report-2017.csv')
# 得到测试和训练数据
train_data=data.sample(frac=0.8)
test_data=data.drop(train_data.index)

input_param_name='Economy..GDP.per.Capita.'
output_param_name='Happiness.Score'

x_train=train_data[[input_param_name]].values
y_train=train_data[[output_param_name]].values

x_test=train_data[[input_param_name]].values
y_test=train_data[[output_param_name]].values

plt.scatter(x_train,y_train,label='Train data',color='red')
plt.scatter(x_test,y_test,label='Test data',color='green')

plt.xlabel(input_param_name)
plt.ylabel(output_param_name)

plt.title('Happy')
plt.legend()
plt.show()


num_iteration=500
learning_rate=0.01
linear_regression=LinearRegression(x_train,y_train)
(theta,cost_history)=linear_regression.train(learning_rate,num_iteration)

print('开始时的损失',cost_history[0])
print('训练后的损失',cost_history[-1])


plt.plot(range(num_iteration),cost_history)
plt.xlabel('iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()


predictions_num=100
x_prediction=np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
y_prediction=linear_regression.predict(x_prediction)

plt.scatter(x_train,y_train,label='Train data',color='red')
plt.scatter(x_test,y_test,label='Test data',color='green')

plt.plot(x_prediction,y_prediction,'r',label='Prediction')

plt.xlabel(input_param_name)
plt.ylabel(output_param_name)

plt.title('Happy')
plt.legend()
plt.show()
