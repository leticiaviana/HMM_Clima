# HMM Clima

Este repositório contém uma implementação simples de um Modelo Oculto de Markov (HMM) para modelar o clima com base em atividades observadas. O exemplo inclui estados ocultos como tipos de clima (ensolarado, nublado, chuvoso) e observações como atividades (caminhada, compras, limpeza).

## Estrutura do Projeto

- **data**: Contém o arquivo `observations.csv` com dados de observação.
- **src**: Contém o código fonte da implementação do HMM.
  - `hmm.py`: Implementação das classes e métodos do HMM.
  - `main.py`: Script principal para rodar o exemplo.
- `README.md`: Este arquivo, explicando o projeto.
- `requirements.txt`: Dependências do projeto.

## Requisitos

- Python 3.x
- Bibliotecas: numpy, pandas

Para instalar as dependências, execute:

```bash
pip install -r requirements.txt
```

## Uso

1. Coloque suas observações no arquivo data/observations.csv.
2. Execute o script principal:

```bash
python src/main.py
```

## Estrutura dos Dados
O arquivo observations.csv deve conter as atividades observadas em cada dia, no seguinte formato:

```bash
dia,atividade
1,caminhada
2,compras
3,limpeza
4,caminhada
5,limpeza
6,compras

```
## Implementação

- **Passos do Algoritmo HMM**:

1. Definição dos Componentes do HMM

```bash
import numpy as np

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

```

2. Inicialização do HMM

```bash
class HMM:
    def __init__(self, states, observations, start_prob, trans_prob, emit_prob):
        self.states = states
        self.observations = observations
        self.start_prob = start_prob
        self.trans_prob = trans_prob
        self.emit_prob = emit_prob

```

3. Algoritmo de Viterbi para Decodificação

```bash
def viterbi(self, obs_seq):
    n_states = len(self.states)
    n_obs = len(obs_seq)

    viterbi = np.zeros((n_states, n_obs))
    backpointer = np.zeros((n_states, n_obs), dtype=int)

    for s in range(n_states):
        viterbi[s, 0] = self.start_prob[s] * self.emit_prob[s, self.observations.index(obs_seq[0])]
        backpointer[s, 0] = 0

    for t in range(1, n_obs):
        for s in range(n_states):
            trans_prob = viterbi[:, t-1] * self.trans_prob[:, s]
            max_trans_prob = np.max(trans_prob)
            viterbi[s, t] = max_trans_prob * self.emit_prob[s, self.observations.index(obs_seq[t])]
            backpointer[s, t] = np.argmax(trans_prob)

    best_path_prob = np.max(viterbi[:, n_obs-1])
    best_last_state = np.argmax(viterbi[:, n_obs-1])

    best_path = [best_last_state]
    for t in range(n_obs-1, 0, -1):
        best_last_state = backpointer[best_last_state, t]
        best_path.insert(0, best_last_state)

    best_path_states = [self.states[state] for state in best_path]

    return best_path_states, best_path_prob

```

## Uso do HMM no Script Principal

1. Carregar Dados de Observação

```bash
import pandas as pd

df = pd.read_csv('data/observations.csv')
observations = df['atividade'].tolist()

```

2. Criar e Treinar o HMM

```bash
hmm = HMM(states, observations_set, start_prob, trans_prob, emit_prob)
best_path_states, best_path_prob = hmm.viterbi(observations)


```

3. Exibir Resultados

```bash
print(f"Observações: {observations}")
print(f"Estados ocultos mais prováveis: {best_path_states}")
```

## Exemplo de Resultados

Após rodar o script principal, você verá a sequência de estados climáticos mais provável para as atividades observadas.

```bash
Observações: ['caminhada', 'compras', 'limpeza']
Estados ocultos mais prováveis: ['ensolarado', 'nublado', 'chuvoso']

```