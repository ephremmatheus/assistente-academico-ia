"""Coleta parágrafos da Wikipédia em português, rotulados por matéria.

Para cada matéria há uma lista de artigos. O texto de cada artigo é baixado
pela API pública da Wikipédia, dividido em parágrafos, e os parágrafos curtos
ou de seções sem conteúdo (Referências, Ver também...) são descartados.
O rótulo de cada parágrafo é a matéria do artigo de onde ele veio.

Licença dos textos: CC BY-SA 4.0 (Wikipédia).

Uso:
    python -m src.machine_learning.coletar_dados
"""

import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from src.caminhos import DADOS

SAIDA = DADOS / "wikipedia_materias.csv"
CACHE = DADOS / "cache_wikipedia"
API = "https://pt.wikipedia.org/w/api.php"
CABECALHO = {"User-Agent": "assistente-academico-ia/1.0 (projeto academico)"}

ARTIGOS = {
    "Biologia": [
        "Célula", "Fotossíntese", "Mitose", "Meiose", "DNA", "Genética", "Evolução", "Seleção natural",
        "Ecossistema", "Sistema imunitário", "Respiração celular", "Bactéria", "Vírus", "Proteína",
        "Enzima", "Cadeia alimentar", "Membrana plasmática", "Cromossomo", "Mutação genética", "Biodiversidade",
    ],
    "História": [
        "Revolução Industrial", "Revolução Francesa", "Independência do Brasil", "Império Romano",
        "Idade Média", "Feudalismo", "Renascimento", "Primeira Guerra Mundial", "Segunda Guerra Mundial",
        "Guerra Fria", "Brasil Colônia", "Escravidão no Brasil", "Proclamação da República do Brasil",
        "Era Vargas", "Ditadura militar no Brasil", "Grandes navegações", "Reforma Protestante",
        "Iluminismo", "Revolução Russa", "Grécia Antiga",
    ],
    "Matemática": [
        "Teorema de Pitágoras", "Função afim", "Função quadrática", "Equação quadrática", "Trigonometria",
        "Logaritmo", "Progressão aritmética", "Progressão geométrica", "Matriz (matemática)", "Probabilidade",
        "Geometria analítica", "Número primo", "Fração", "Derivada", "Integral", "Polinômio", "Triângulo",
        "Círculo", "Conjunto", "Número real",
    ],
    "Química": [
        "Tabela periódica", "Átomo", "Ligação química", "Ligação covalente", "Ligação iônica", "Ácido",
        "Base (química)", "Sal (química)", "PH", "Reação química", "Estequiometria", "Química orgânica",
        "Hidrocarboneto", "Oxirredução", "Eletroquímica", "Mol", "Isótopo", "Elemento químico",
        "Solução (química)", "Termoquímica",
    ],
    "Geografia": [
        "Clima semiárido", "Caatinga", "Tectônica de placas", "Relevo", "Bacia hidrográfica",
        "Clima tropical", "Floresta Amazônica", "Cerrado", "Mata Atlântica", "Urbanização", "Globalização",
        "Demografia", "Efeito estufa", "Aquecimento global", "Vulcão", "Sismo", "Rio São Francisco",
        "Cartografia", "Latitude", "Clima equatorial",
    ],
    "Programação": [
        "Algoritmo", "Linguagem de programação", "Python", "Estrutura de dados", "Busca em largura",
        "Busca em profundidade", "Recursividade (ciência da computação)", "Variável (programação)",
        "Programação orientada a objetos", "Algoritmo de ordenação", "Lista ligada", "Pilha (informática)",
        "Fila (estrutura de dados)", "Árvore (estrutura de dados)", "Compilador", "Complexidade computacional",
        "Banco de dados", "Algoritmo de Dijkstra", "Java (linguagem de programação)", "Estrutura de repetição",
    ],
}

SECOES_DESCARTADAS = {"referências", "ver também", "ligações externas", "bibliografia", "notas",
                      "leitura adicional", "fontes", "notas e referências"}
MINIMO_PALAVRAS = 30
MAXIMO_PARAGRAFOS_POR_ARTIGO = 15


def baixar_texto(titulo: str, tentativas: int = 6) -> str:
    """Baixa o texto puro de um artigo, guardando em cache local.

    A API devolve erro 429 quando recebe pedidos rápido demais; nesse caso
    esperamos cada vez mais antes de tentar de novo.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    arquivo_cache = CACHE / f"{hashlib.md5(titulo.encode('utf-8')).hexdigest()}.txt"
    if arquivo_cache.exists():
        return arquivo_cache.read_text(encoding="utf-8")

    parametros = {
        "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
        "titles": titulo, "format": "json", "formatversion": 2,
    }
    url = f"{API}?{urllib.parse.urlencode(parametros)}"
    requisicao = urllib.request.Request(url, headers=CABECALHO)
    for tentativa in range(1, tentativas + 1):
        try:
            with urllib.request.urlopen(requisicao, timeout=30) as resposta:
                dados = json.load(resposta)
            break
        except urllib.error.HTTPError as erro:
            if erro.code != 429 or tentativa == tentativas:
                raise
            espera = int(erro.headers.get("Retry-After") or 0) or 5 * tentativa
            time.sleep(espera)
    pagina = dados["query"]["pages"][0]
    texto = pagina.get("extract", "") if not pagina.get("missing") else ""
    arquivo_cache.write_text(texto, encoding="utf-8")
    return texto


def extrair_paragrafos(texto: str) -> list[str]:
    paragrafos, secao = [], ""
    for bloco in texto.split("\n"):
        bloco = bloco.strip()
        titulo_secao = re.fullmatch(r"=+\s*(.*?)\s*=+", bloco)
        if titulo_secao:
            secao = titulo_secao.group(1).lower()
            continue
        if secao in SECOES_DESCARTADAS or len(bloco.split()) < MINIMO_PALAVRAS:
            continue
        paragrafos.append(re.sub(r"\s+", " ", bloco))
    return paragrafos[:MAXIMO_PARAGRAFOS_POR_ARTIGO]


def main() -> None:
    linhas = []
    for materia, titulos in ARTIGOS.items():
        for titulo in titulos:
            try:
                paragrafos = extrair_paragrafos(baixar_texto(titulo))
            except Exception as erro:  # rede instável não deve derrubar a coleta inteira
                print(f"  [erro] {titulo}: {erro}")
                continue
            if not paragrafos:
                print(f"  [vazio] {titulo}")
            linhas += [{"materia": materia, "artigo": titulo, "texto": p} for p in paragrafos]
            time.sleep(1.0)  # respeitar o limite de pedidos da API
        print(f"{materia}: {sum(1 for l in linhas if l['materia'] == materia)} parágrafos")

    with SAIDA.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=["materia", "artigo", "texto"])
        escritor.writeheader()
        escritor.writerows(linhas)
    print(f"\n{len(linhas)} parágrafos salvos em {SAIDA}")


if __name__ == "__main__":
    main()
