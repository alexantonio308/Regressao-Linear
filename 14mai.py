import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import seaborn as seabornInstance 
from sklearn import metrics

import matplotlib.pyplot as plt

#Leitura do Arquivo de dados
base = pd.read_csv("C:/Users/alexa/Desktop/I.A/items.csv", sep=',')
print("Número de linhas/Colunas iniciais: \n", base.shape)

#Excluir linhas com dados Nan
base.dropna(inplace=True)
print("Número de linhas/Colunas removendo Nan: \n", base.shape)

#Informações 
print("\nDescrição da tabela :\n",base.describe())

#Adicionar dados em arrays
rm = np.array(base["rating"]).reshape(-1,1)
medv = np.array(base["totalReviews"]).reshape(-1,1)


#Test_size = 20% de taxa de treino
X_train, X_test, y_train, y_test = train_test_split(rm, medv, test_size=0.2, random_state=0)

#Treinamento
linreg= LinearRegression()
linreg.fit(X_train, y_train)
linreg.score(X_test,y_test)

#Valor de inclinação
print('\nCoeficiente de Inclinação (w): {}'.format(linreg.coef_))
 #Valores de interceptação
print(list(map('Linha de Interceptação (b): {:.3f}'.format,linreg.intercept_)))
 
print('R-squared score (treinamento): {:.3f}'.format(linreg.score(X_train, y_train)))
 
print('R-squared score (test): {:.3f}\n\n'.format(linreg.score(X_test, y_test)))

#Previsões
y_pred = linreg.predict(X_test)
df = pd.DataFrame({'Atual': y_test.flatten(), 'Previsão': y_pred.flatten()})
print(df)

#Erro médio
print('\nErro médio Quadrático:', metrics.mean_squared_error(y_test, y_pred)) 
#Plotagem dos Gráficos
#Gráfico com dados de treinamento
plt.scatter(X_train, y_train, marker='D', s=40, alpha=0.9)
plt.plot(X_train, linreg.coef_ * X_train + linreg.intercept_, 'r-')
plt.title('Preço de Produtos')
plt.xlabel('Rating (x)')
plt.ylabel('totalReviews (y)')
plt.show()

#Gráfico com dados de teste
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Preço de Produtos')
plt.xlabel('Rating (x)')
plt.ylabel('totalReviews (y)')
plt.show()

 
