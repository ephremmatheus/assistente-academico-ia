"""Agente de decisão baseado em regras.

Recebe o que os outros módulos produziram (texto e confiança do OCR, matéria
prevista, intenção da pergunta, trechos encontrados e suas similaridades) e
decide a ação:

- confiança do OCR baixa       -> pedir uma foto melhor
- intenção "resumo"            -> resumir o material
- intenção "definição"/factual -> responder com QA extrativo
- similaridade abaixo do limite -> responder "não encontrei isso no material"
- matéria da pergunta diferente da imagem -> avisar o usuário

Os limites são heurísticas e serão calibrados nos experimentos.
"""
