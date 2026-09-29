# Decisão de campanha e métricas

## Por que accuracy não é suficiente

Na base histórica, apenas 707 de 5.000 clientes pertencem à classe churn
(14,14%). Um classificador que sempre respondesse `não churn` teria cerca de
85,86% de accuracy sem identificar nenhum cliente em risco.

Por isso, accuracy aparece apenas como métrica complementar. O foco operacional
é a capacidade de ordenar clientes por risco e definir uma ação compatível com
o custo da campanha.

## Benchmark atual no limiar 0,50

| Resultado | Quantidade |
|---|---:|
| Verdadeiro positivo (TP) | 103 |
| Falso positivo (FP) | 9 |
| Falso negativo (FN) | 38 |
| Verdadeiro negativo (TN) | 850 |

Isso produz precision de 0,920: aproximadamente 92% dos clientes indicados
realmente pertenciam à classe churn. O recall de 0,730 identifica cerca de 73%
dos clientes churn no conjunto de teste.

## Qual métrica priorizar?

- **Campanha cara ou capacidade limitada:** priorizar precision e usar os
  clientes de maior probabilidade. Acompanhar `precision@k`, em que `k` é o
  número máximo de contatos possíveis.
- **Ação barata e perda de cliente cara:** priorizar recall, aceitando mais
  falsos positivos para reduzir clientes churn não abordados.
- **Sem custos definidos:** não escolher um limiar definitivo. Reportar a curva
  precision-recall, PR-AUC, `precision@k` e `recall@k` antes da decisão.

## Regra para versões futuras

O limiar 0,50 é apenas o baseline técnico. Uma versão voltada a negócio deve
registrar o orçamento/capacidade da campanha e o custo estimado de FP e FN;
então seleciona o limiar no conjunto de validação, nunca no conjunto de teste.
