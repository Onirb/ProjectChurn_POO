# Dados locais

Nenhuma base de clientes ou modelo treinado é versionado neste repositório.

## Contrato aceito

O projeto opera com 17 variáveis preditoras e o alvo `churn`. A validação aceita
nomes com espaço, hífen ou sublinhado e converte-os para o padrão interno. As
colunas opcionais `state`, `area code` e `phone number` são descartadas antes
do treino, pois não fazem parte do contrato da API. Valores ausentes nas
variáveis preditoras são preservados e contabilizados: a política de imputação
fica registrada no pipeline de treino, e não escondida durante a aquisição.

## Fontes candidatas

1. **Legado de 5.000 linhas** — melhor candidato para recuperar os resultados
   históricos descritos no repositório. O esquema coincide com o conjunto
   [`CHURN` do pacote regclass](https://search.r-project.org/CRAN/refmans/regclass/html/CHURN.html),
   mas a origem exata ainda deve ser confirmada antes de qualquer alegação de
   reprodução exata.
2. **[Telecom Churn Dataset (Kaggle / BigML)](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)** — benchmark alternativo público.
   Sua variante conhecida possui 3.333 linhas ao combinar as partições 80/20;
   resultados obtidos com ela devem ser identificados como um novo benchmark.

## Preparação

Após obter manualmente um CSV de uma fonte permitida, execute a partir da raiz:

```powershell
python -m scripts.prepare_data caminho\\para\\arquivo.csv
```

O comando cria `data/processed/churn.csv` e `dataset_metadata.json`, ambos
locais e ignorados pelo Git. O treino deve consumir apenas essa cópia canônica.

## Lote sintético de demonstração

`data/demo/customer_batch_demo.csv` é um lote pequeno e sintético, versionado
somente para demonstrar o fluxo de pontuação de campanha. Ele não contém dados
de clientes reais e não é usado para treinar ou medir modelos.
