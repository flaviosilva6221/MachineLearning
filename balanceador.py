import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style = 'white')
sns.set(style = 'whitegrid', color_codes = True)

dados = pd.read_csv('bank.csv',sep=',')
print(dados.shape) # shape retorna o numero de linhas e de colunas do dataframe
print(list(dados.columns)) # imprime os rotulos das colunas
print(dados.head())

print('education antes de alterar')
print(dados['education'].unique())

# Alterar os dados basic.4y, basic.6y, basic.9y para basic
dados['education'] = np.where(dados['education'] == 'basic.4y','basic',dados['education'])
dados['education'] = np.where(dados['education'] == 'basic.6y','basic',dados['education'])
dados['education'] = np.where(dados['education'] == 'basic.9y','basic',dados['education'])

# dados['y_'] = ''
# dados['y_']==np.where(dados['education'] == 0,'N',dados['y_'])
# dados['y_']==np.where(dados['education'] == 1,'S',dados['y_'])

print('education após de alterar')
print(dados['education'].unique())

print ('Frequencias de classes')
print(dados['y'].value_counts())

# sns.countplot(x='y',data=dados,palette='hls')
# plt.show()

print(dados.groupby('y').mean())

# Visualização de dados
# pd.crosstab(dados.job,dados.y).plot(kind='bar')
# plt.title('Adesão em função do cargo')
# plt.xlabel('Cargo')
# plt.ylabel('Frequencia da adesão')
# plt.savefig('Adesao_cargo')
# plt.show()

# Normalizar variáveis categóricas
atributos_categoricos = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']

for att in atributos_categoricos:
    lista_cat = pd.get_dummies(dados[att],prefix=att)
    dados1 = dados.join(lista_cat)
    dados1.drop(att,axis=1,inplace=True)
    dados = dados1

#print(list(dados.columns)) # imprime os rotulos das colunas

#print(dados.head())

############################################################################
# Balanceamento dos dados
############################################################################

# Separar atributos de classe
Y = dados['y']
X = dados.drop('y',axis=1,inplace=False)


#SMOTE > SYNTHETIC MINORITY OVERSAMPLING TECHNIC
# Utiliza KNN para determinar instancias sintéticos para as classes minonitárias

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

balanceador = SMOTE(random_state=0) # Objeto que balanceará os dados

#Segmentar os dados em dados para aprendizado e dados para testes
X_train,X_test,Y_Train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
colunas = X_train.columns

#aplicar o SMOTE
X_balanceado, Y_balanceado = balanceador.fit_sample(X_train,Y_Train)

#Recompor dados e rotulos
X_balanceado = pd.DataFrame(data=X_balanceado,columns=colunas)
Y_balanceado = pd.DataFrame(data=Y_balanceado,columns=['y'])

# print(X_balanceado)
# print(Y_balanceado)

################################################################################################
#Efeito do balanceamento
################################################################################################
#Tamanho da nova base
print('Tamanho da base balanceada :', len(X_balanceado))

#Numero de adesões
print('Quantidade adesões :' , len(Y_balanceado[Y_balanceado['y']==1]))

#Numero de recusas
print('Quantidade recusas :' , len(Y_balanceado[Y_balanceado['y']==0]))

X_train = X_balanceado
Y_Train = Y_balanceado


################################################################################################
from sklearn.linear_model import LogisticRegression

#Obter o modelo de Logit
logit = LogisticRegression(random_state=0).fit(X_train,Y_Train)
res = logit.predict(X_test[:3]) # Classificar a terceira linha dos dados X_teste
print(res)

distribuicao_probalistica = logit.predict_proba(X_test[:3])
print(distribuicao_probalistica)

