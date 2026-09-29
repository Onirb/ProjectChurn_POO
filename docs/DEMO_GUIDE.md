# Guia de demonstração local

O objetivo da demo é mostrar um fluxo de portfólio: planejamento comercial,
pontuação de um lote e lista priorizada. Não use dados reais de clientes.

## Preparação

1. Obtenha uma base permitida e execute `python -m scripts.prepare_data`.
2. Gere os artefatos locais de demonstração:

```powershell
python -m scripts.build_portfolio_demo_models
```

3. Use `data/demo/customer_batch_demo.csv` como exemplo de lote. O arquivo é
sintético, contém somente variáveis previstas pelo contrato e um `customer_id`.

4. Abra o painel local:

```powershell
streamlit run streamlit_app.py
```

## Fluxo comercial

1. Consulte as ações disponíveis em `GET /campaign/objectives`.
2. Informe objetivo, capacidade e, se necessário, como preencher uma meta que
ultrapasse o perfil original em `POST /campaign/plan`.
3. Envie os registros para `POST /campaign/score` ou use o futuro painel local.

O painel `streamlit_app.py` já implementa esse fluxo e permite baixar a lista
priorizada em CSV ao fim da demonstração.

O resultado contém apenas identificador, probabilidade, posição no ranking e
indicação de inclusão. Nenhuma decisão é enviada automaticamente a clientes.

## Limites deliberados

- A demo opera localmente e não armazena lotes enviados.
- As escolhas comerciais foram calibradas no benchmark local; não são regras
  para uma campanha real.
- O endpoint `/predict` preserva seu modelo anterior e não é substituído pelos
  artefatos de demonstração.
