# Auditoria: exploração histórica para produto reproduzível

## Escopo e evidência

Esta auditoria compara o notebook histórico `notebooks/Churn.ipynb` do
repositório inicial com a aplicação atual. A base é o arquivo
`data/churn_data.csv` da versão inicial: 5.000 registros e 18 colunas.

O notebook também continha `NLP.ipynb`, removido depois do histórico público.
Ele usa LSTM e `Conv1d` para classificação de variantes de português. Não faz
parte do problema de churn e não é evidência de uma CNN treinada nesta base.

## O que a exploração original fez bem

- Aplicou CRISP-DM: contexto de negócio, EDA, preparação, modelagem e análise
  de métricas.
- Mediu ausência de dados, distribuição da classe, correlações e importância
  preliminar de variáveis.
- Comparou d1--d4 e sete famílias: Random Forest, XGBoost, SVM, KNN, árvore,
  Naive Bayes e MLP.
- Reconheceu que accuracy isolada é inadequada para uma classe positiva de
  aproximadamente 14%.

## Limites que impedem usar o notebook como produção

### Vazamento de pré-processamento

Em d3, a mediana e o `MinMaxScaler` são ajustados no dataframe inteiro antes
da criação das dobras. d1, d2 e d4 também calculam estatísticas globais antes
da separação. Isso não invalida a exploração, mas pode tornar a estimativa
ligeiramente otimista. A versão atual mantém imputação, escala e `log1p`
dentro do `Pipeline`, ajustados apenas no treino de cada dobra.

O CSV recuperado tem 50 ausências também em `internationalplan` e
`voicemailplan`. O código `fillna(d3.median())` do notebook não imputava essas
colunas categóricas. A reconstrução completa-as com a moda global, marcada
como aproximação explícita; a variante sem vazamento aprende a moda em cada
treino.

### Variações não apropriadas para o serviço

- d1 descarta linhas e muda a população avaliada.
- d2 substitui outliers com regras globais; deve ser estudado como ablação,
  não assumido como padrão.
- d4 aplica Savitzky--Golay em uma tabela sem ordem temporal; ao suavizar uma
  linha com informação das vizinhas, introduz uma dependência artificial da
  ordem do CSV.

### Tuning e teste

O notebook ajusta hiperparâmetros na primeira divisão externa e depois reporta
resultados em várias divisões. A aplicação atual separa seleção por validação
cruzada e um holdout final. O experimento `controlled_tabular_v1` registra
família, hiperparâmetro e limiar sem alterar a API.

## Estado atual de engenharia e DevOps

### Já presente

- contrato canônico de 17 features;
- pipeline único de treino/inferência;
- API FastAPI, logs e endpoint de métricas simples;
- testes de contrato, pipeline e API;
- rastreamento local com MLflow;
- Dockerfile e dependências fixadas.

### Pendências antes de chamar de implantação

1. Alinhar a versão do Python do Dockerfile com o ambiente testado (3.12.10).
2. Criar CI para testes, checagem de imagem e auditoria de dependências.
3. Versionar o artefato, dados, limiar e configuração do modelo promovido.
4. Expor métricas de serviço com armazenamento/monitoramento externo, em vez
   de listas em memória.
5. Definir política de retenção: capacidade, custo de falso positivo e falso
   negativo, e limiar resultante.
6. Validar a imagem com Docker Desktop ativo antes de qualquer publicação.

## Protocolo de reconstrução d3

`python -m scripts.run_legacy_d3_comparison` executa as mesmas cinco divisões
estratificadas do notebook. Ele compara:

- `historical_global_d3`: reprodução para auditoria, com estatísticas globais;
- `leakage_free_d3`: imputação, Min-Max e `log1p` aprendidos no treino.

O XGBoost e um MLP leve com parada antecipada foram recolocados também no
`controlled_tabular_v1`, no qual competem com Gradient Boosting, Random
Forest, Extra Trees e regressão logística sob um holdout separado e a mesma
política de seleção de limiar.

Os resultados são locais em `runs/legacy_d3_reconstruction_v1/`, não atualizam
o artefato da API e não devem ser comparados diretamente com o holdout do
experimento controlado, que usa outro protocolo.

O `mlp_baseline` é uma implementação atual com parada antecipada para manter o
tempo de execução controlado. Não substitui nem afirma reproduzir exatamente o
`MLPClassifier(max_iter=500)` do notebook.
