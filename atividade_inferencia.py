from sklearn.cluster import KMeans # indutor
from sklearn.preprocessing import StandardScaler # Para Normalizar
import joblib # para salvar e carregar o modelo salvo
from numpy.random import randint

#carrega o modelo salvo
kmeans_salvo = joblib.load('cluster.model')

#Gera dados da nova instancia
values = randint(0, 10, 20)
#print(values)

# dados da nova instancia
nova_instancia = [values]
print(nova_instancia)

# Normalizados os dados
normalizador =  joblib.load('normalizador.save') 
dados_normalizados = normalizador.transform(nova_instancia)

print('indice do cluster previsto para a nova instancia', kmeans_salvo.predict(dados_normalizados))