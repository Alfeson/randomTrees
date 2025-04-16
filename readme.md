# 🚀 Spaceship Titanic - Classificação com Random Forest Customizada

Este projeto foi desenvolvido para resolver o desafio do **Spaceship Titanic**, inspirado no famoso desastre marítimo, mas ambientado em um cenário futurista, intergaláctico e com um toque de ficção científica.

## 🧩 Descrição do Problema

No ano 2912, a nave interestelar **Spaceship Titanic** estava em sua viagem inaugural com destino a exoplanetas habitáveis. No entanto, ao passar por Alpha Centauri, a nave colidiu com uma anomalia espaço-temporal, resultando no desaparecimento misterioso de **metade dos passageiros**, transportados para outra dimensão.

Seu papel como cientista de dados é ajudar a determinar **quais passageiros foram transportados** com base nos dados resgatados da nave, utilizando aprendizado de máquina para **classificar corretamente** cada um.

## 🧠 Solução Implementada

A solução proposta neste repositório é uma **implementação do zero de uma Random Forest personalizada** (sem uso direto do `RandomForestClassifier` do Scikit-Learn). A abordagem consiste em:

- Utilizar **bootstrap** (amostragem com reposição) para treinar múltiplas árvores de decisão.
- Treinar cada árvore em um subconjunto aleatório das colunas (features).
- Utilizar **votação majoritária** para decidir a predição final de cada passageiro.

Essa abordagem visa não apenas resolver o problema com boa performance, mas também **compreender a fundo o funcionamento interno de um ensemble de árvores**.

## 📁 Estrutura do Projeto

- `random_forest_custom.py`: Script principal contendo a implementação da classe `RandomForest`, o pipeline de treino/validação e geração de submissão.
- `train.csv`: Conjunto de dados com passageiros rotulados (utilizado para treinar e validar o modelo).
- `test.csv`: Conjunto de dados sem rótulo, usado para gerar a submissão final.
- `submission_random_forest_custom.csv`: Arquivo final contendo as predições para envio na competição.

## ▶️ Como Executar

1. Certifique-se de ter os arquivos `train.csv` e `test.csv` na mesma pasta que o script.
2. Execute o comando `pip install -r requeriments` para baixar as dependências do projeto
2. Execute o script com Python 3:

```bash
python random_forest_custom.py
