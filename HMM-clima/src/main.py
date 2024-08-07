import pandas as pd
from hmm import HMM

# Carregar os dados de observação
df = pd.read_csv('data/observations.csv')
observations = df['atividade'].tolist()

# Definir os estados, observações e probabilidades
states = ['ensolarado', 'nublado', 'chuvoso']
observations_set = ['caminhada', 'compras', 'limpeza']
start_prob = np.array([0.6, 0.3, 0.1])
trans_prob = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])
emit_prob = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.4, 0.4],
    [0.1, 0.3, 0.6]
])

# Criar e treinar o HMM
hmm = HMM(states, observations_set, start_prob, trans_prob, emit_prob)
best_path_states, best_path_prob = hmm.viterbi(observations)

print(f"Observações: {observations}")
print(f"Estados ocultos mais prováveis: {best_path_states}")
