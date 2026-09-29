# Proveniência e versões dos dados

## Regra de publicação

Métricas só podem ser chamadas de reproduzidas quando a fonte, o número de
linhas, o esquema e a divisão treino/teste forem registrados junto ao experimento.

## Estado histórico

O README do projeto anterior descreve 5.000 observações e 18 colunas, incluindo
o alvo. A cópia recuperada em `Onirb/ProjectChurn`, arquivo
`data/churn_data.csv`, confirma esse esquema e possui a assinatura SHA-256
`934ba6d89d7af70606af9153e0c998f4fde1c2b39745547bafafb2b053c2ea91`.

Ela é a referência local para o Track A de reprodutibilidade. As métricas
antigas continuam identificadas como históricas até serem novamente executadas
com seed, dependências e resultados registrados.

## Benchmark alternativo

O conjunto Telecom Churn / BigML, distribuído no Kaggle, é compatível após a
remoção de colunas contextuais. Como possui outra amostragem, deve originar uma
execução nova, identificada pelo seu próprio tamanho amostral e métricas.

## Registro mínimo por execução

- URL ou identificador da fonte;
- data de aquisição e licença/termos aplicáveis;
- hash do CSV bruto;
- número de linhas após validação;
- seed da divisão treino/teste;
- versão do código e dependências;
- métricas e matriz de confusão.
