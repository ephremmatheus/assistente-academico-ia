"""Processamento de Linguagem Natural.

Planejado:
- intencao.py: reconhecer o tipo de pergunta (definição, resumo, lista, factual).
- busca.py: dividir o texto extraído em trechos e encontrar os mais relevantes
  para a pergunta (TF-IDF como base de comparação e embeddings de sentenças).
- resposta.py: extrair a resposta do trecho encontrado (QA extrativo) ou resumir.

Avaliação: acurácia da intenção; Recall@k e MRR da busca; Exact Match e F1 das respostas.
"""
