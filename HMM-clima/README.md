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