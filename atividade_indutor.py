import pandas as pd
from sklearn.cluster import KMeans # indutor
from sklearn.preprocessing import StandardScaler # Para Normalizar
import joblib # para salvar o modelo
from kneed import KneeLocator # para determinar o numero ideal de cluster

#Lê os dados do csv
dados = pd.read_csv('Absenteeism_at_work.csv',sep = ';')
#dropa a coluna ID
dados.drop('ID',axis='columns', inplace=True)

# Normalizados os dados
normalizador = StandardScaler()
dados_normalizados = normalizador.fit_transform(dados)

joblib.dump(normalizador,'normalizador.save')

# Determinar o número ideal de clusters
# Elbow (cotovelo)
kmeans_kwargs = {
    'init': 'random',
    'n_init': 10,
    'max_iter': 300,
    'random_state': 42
}

# Lista de SSE para cada K, onde K será uma iteração do método para avaliação
sse = []

# Navegar em um vetor com um numero de cluster a serem avaliados
for k in range(1,100):
    kmeans = KMeans(n_clusters=k,**kmeans_kwargs)
    kmeans.fit(dados_normalizados)
    sse.append(kmeans.inertia_)


kl = KneeLocator(
    range(1,100),
    sse,
    curve='convex',
    direction='decreasing'
)
print('numero ideal de clusters: ', kl.elbow)

# Obter o cluster
kmeans = KMeans(
    init = 'random',
    n_clusters = kl.elbow, # número de agrupamentos(clusters) a serem obtidos
    n_init = 10, # número de inicializações para a obtenção da convergência
    max_iter = 300, # número máximo de iterações por inicialização
    random_state = 42 # fator para aleatoriedade de seleção das instacia
)

# obtenção do modelo de cluster
kmeans_model = kmeans.fit(dados_normalizados) 

print('Centroides')
print(kmeans.cluster_centers_)

# Salvar o modelo em disco para uso posterior
joblib.dump(kmeans_model,'cluster.model')
