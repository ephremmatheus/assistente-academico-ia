"""Experimento 1: o pré-processamento melhora o OCR?

Para cada imagem com gabarito, o OCR roda com quatro configurações de
pré-processamento e o resultado é comparado ao texto correto (CER e WER).

Saídas (o prefixo inclui o nome do motor, ex.: "ocr_easyocr"):
    resultados/<prefixo>_por_imagem.csv          uma linha por imagem x configuração
    resultados/<prefixo>_resumo.csv              médias por configuração
    resultados/graficos/<prefixo>_cer_por_configuracao.png
    resultados/graficos/<prefixo>_cer_vs_inclinacao.png   (só para as sintéticas)
    resultados/exemplos/etapas_preprocessamento.png
    resultados/exemplos/<prefixo>_exemplo.txt

Uso:
    python -m src.visao_computacional.experimento_ocr [--motor easyocr] [--limite 30] [--pasta dados/imagens_reais]

    # Refazer só os gráficos a partir do CSV já salvo (sem rodar o OCR de novo)
    python -m src.visao_computacional.experimento_ocr --so-graficos
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.caminhos import EXEMPLOS, GRAFICOS, IMAGENS_SINTETICAS, RESULTADOS  # noqa: E402

from .avaliacao import cer, wer  # noqa: E402
from .gerar_imagens_sinteticas import gerar  # noqa: E402
from .io_imagem import carregar_imagem  # noqa: E402
from .ocr import criar_motor  # noqa: E402
from .preprocessamento import ConfigPreprocessamento, preprocessar  # noqa: E402

CONFIGURACOES: dict[str, ConfigPreprocessamento | None] = {
    "Sem pré-processamento": None,
    "Iluminação + inclinação": ConfigPreprocessamento(remover_ruido=False),
    "Completo (+ remoção de ruído)": ConfigPreprocessamento(),
    "Completo + binarização": ConfigPreprocessamento(binarizar=True),
}

# Cores e tinta dos gráficos: uma cor por série, texto sempre em tons neutros.
COR_SERIE_1 = "#2a78d6"
COR_SERIE_2 = "#eb6834"
COR_TEXTO = "#0b0b0b"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_GRADE = "#e4e3df"


def listar_pares(pasta: Path) -> list[tuple[Path, Path]]:
    """Encontra imagens que têm um .txt com o mesmo nome (o gabarito)."""
    pares = []
    for imagem in sorted(pasta.iterdir()):
        if imagem.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            gabarito = imagem.with_suffix(".txt")
            if gabarito.exists():
                pares.append((imagem, gabarito))
    return pares


def rodar(pasta: Path, nome_motor: str, limite: int | None) -> pd.DataFrame:
    motor = criar_motor(nome_motor)
    pares = listar_pares(pasta)
    if limite:
        pares = pares[:limite]
    if not pares:
        raise SystemExit(f"Nenhuma imagem com gabarito encontrada em {pasta}")

    linhas = []
    for n, (caminho_imagem, caminho_gabarito) in enumerate(pares, start=1):
        imagem = carregar_imagem(caminho_imagem)
        gabarito = caminho_gabarito.read_text(encoding="utf-8")
        for nome_config, config in CONFIGURACOES.items():
            inicio = time.perf_counter()
            angulo = 0.0
            if config is None:
                entrada = imagem
            else:
                preparado = preprocessar(imagem, config)
                entrada, angulo = preparado.imagem, preparado.angulo_corrigido
            resultado = motor.reconhecer(entrada)
            duracao = time.perf_counter() - inicio
            linhas.append({
                "arquivo": caminho_imagem.name,
                "configuracao": nome_config,
                "cer": cer(gabarito, resultado.texto),
                "wer": wer(gabarito, resultado.texto),
                "confianca_media": resultado.confianca_media,
                "angulo_corrigido": angulo,
                "tempo_s": duracao,
                "texto_ocr": resultado.texto,
            })
        desta_imagem = linhas[-len(CONFIGURACOES):]
        resumo = " | ".join(f"{linha['cer']:.3f}" for linha in desta_imagem)
        print(f"[{n}/{len(pares)}] {caminho_imagem.name}  CER por configuração: {resumo}", flush=True)
    return pd.DataFrame(linhas)


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    resumo = df.groupby("configuracao", sort=False).agg(
        imagens=("arquivo", "count"),
        cer_medio=("cer", "mean"),
        cer_mediana=("cer", "median"),
        cer_desvio=("cer", "std"),
        wer_medio=("wer", "mean"),
        confianca_media=("confianca_media", "mean"),
        tempo_medio_s=("tempo_s", "mean"),
    )
    return resumo.reindex(list(CONFIGURACOES)).reset_index()


def _estilizar_eixos(ax) -> None:
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(COR_TEXTO_SECUNDARIO)
    ax.tick_params(colors=COR_TEXTO_SECUNDARIO, length=0)
    ax.grid(axis="x", color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def grafico_cer_por_configuracao(resumo: pd.DataFrame, motor: str, destino: Path) -> None:
    """Barras horizontais com o CER médio e o intervalo de 95% (± 1,96 erro padrão)."""
    dados = resumo.iloc[::-1]  # a primeira configuração fica no topo
    cer_pct = dados["cer_medio"] * 100
    erro_pct = 1.96 * dados["cer_desvio"].fillna(0) / dados["imagens"] ** 0.5 * 100

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    ax.barh(dados["configuracao"], cer_pct, height=0.55, color=COR_SERIE_1)
    ax.errorbar(cer_pct, dados["configuracao"], xerr=erro_pct, fmt="none",
                ecolor=COR_TEXTO_SECUNDARIO, elinewidth=1, capsize=3)
    for y, (valor, erro) in enumerate(zip(cer_pct, erro_pct)):
        ax.text(valor + erro + 0.4, y, f"{valor:.1f}%", va="center", fontsize=9, color=COR_TEXTO)

    _estilizar_eixos(ax)
    ax.tick_params(axis="y", labelcolor=COR_TEXTO, labelsize=9)
    ax.set_xlabel("CER médio (%) · menor é melhor", color=COR_TEXTO_SECUNDARIO, fontsize=9)
    ax.set_xlim(0, (cer_pct + erro_pct).max() * 1.2 + 1)
    n = int(dados["imagens"].iloc[0])
    ax.set_title(f"CER médio por configuração ({motor}, {n} imagens)", loc="left", fontsize=10, color=COR_TEXTO)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, facecolor="white")
    plt.close(fig)


def grafico_cer_vs_inclinacao(df: pd.DataFrame, metadados: pd.DataFrame, motor: str, destino: Path) -> None:
    """Dispersão do CER em função da inclinação da imagem, com e sem pré-processamento."""
    juntos = df.merge(metadados[["arquivo", "angulo"]], on="arquivo")
    juntos["inclinacao_abs"] = juntos["angulo"].abs()
    series = [("Sem pré-processamento", COR_SERIE_1), ("Completo (+ remoção de ruído)", COR_SERIE_2)]

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    for nome, cor in series:
        parte = juntos[juntos["configuracao"] == nome]
        ax.scatter(parte["inclinacao_abs"], parte["cer"] * 100, s=28, color=cor,
                   edgecolors="white", linewidths=1, label=nome)

    _estilizar_eixos(ax)
    ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Inclinação da imagem (graus, valor absoluto)", color=COR_TEXTO_SECUNDARIO, fontsize=9)
    ax.set_ylabel("CER (%)", color=COR_TEXTO_SECUNDARIO, fontsize=9)
    ax.set_title(f"Efeito da inclinação no erro do OCR ({motor})", loc="left", fontsize=10, color=COR_TEXTO)
    legenda = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for texto in legenda.get_texts():
        texto.set_color(COR_TEXTO)
    fig.tight_layout()
    fig.savefig(destino, facecolor="white")
    plt.close(fig)


def figura_etapas(caminho_imagem: Path, destino: Path) -> None:
    """Mostra a mesma imagem em cada etapa do pré-processamento (figura para o artigo)."""
    preparado = preprocessar(carregar_imagem(caminho_imagem), ConfigPreprocessamento(binarizar=True))
    titulos = {
        "original": "Original",
        "sem_ruido": "Ruído removido",
        "iluminacao_corrigida": "Iluminação corrigida",
        "inclinacao_corrigida": f"Inclinação corrigida ({preparado.angulo_corrigido:+.1f}°)",
        "binarizada": "Binarizada",
    }
    etapas = [(titulos[k], preparado.etapas[k]) for k in titulos if k in preparado.etapas]

    fig, eixos = plt.subplots(len(etapas), 1, figsize=(7, 2.1 * len(etapas)), dpi=150)
    for ax, (titulo, imagem) in zip(eixos, etapas):
        if imagem.ndim == 3:
            imagem = imagem[:, :, ::-1]  # BGR -> RGB para o matplotlib
        ax.imshow(imagem, cmap="gray", vmin=0, vmax=255)
        ax.set_title(titulo, loc="left", fontsize=9, color=COR_TEXTO)
        ax.axis("off")
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, facecolor="white")
    plt.close(fig)


def salvar_exemplo(df: pd.DataFrame, pasta: Path, destino: Path) -> None:
    """Grava, para a primeira imagem, o gabarito e o texto lido em cada configuração."""
    arquivo = df["arquivo"].iloc[0]
    gabarito = (pasta / arquivo).with_suffix(".txt").read_text(encoding="utf-8")
    partes = [f"Imagem: {arquivo}\n", "=== Gabarito ===", gabarito, ""]
    for _, linha in df[df["arquivo"] == arquivo].iterrows():
        partes += [f"=== {linha['configuracao']} (CER {linha['cer']:.3f}, WER {linha['wer']:.3f}) ===",
                   linha["texto_ocr"], ""]
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(partes), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara configurações de pré-processamento no OCR.")
    parser.add_argument("--pasta", type=Path, default=IMAGENS_SINTETICAS)
    parser.add_argument("--motor", choices=["easyocr", "tesseract"], default="easyocr")
    parser.add_argument("--limite", type=int, default=None, help="usar só as N primeiras imagens")
    parser.add_argument("--so-graficos", action="store_true",
                        help="refaz os gráficos a partir do CSV salvo, sem rodar o OCR")
    args = parser.parse_args()

    if args.pasta == IMAGENS_SINTETICAS and not any(IMAGENS_SINTETICAS.glob("*.jpg")):
        print("Gerando as imagens sintéticas (semente 42)...")
        gerar(quantidade=60, semente=42)

    prefixo = f"ocr_{args.motor}" + ("" if args.pasta == IMAGENS_SINTETICAS else f"_{args.pasta.name}")
    caminho_csv = RESULTADOS / f"{prefixo}_por_imagem.csv"

    if args.so_graficos:
        df = pd.read_csv(caminho_csv, encoding="utf-8", keep_default_na=False)
    else:
        df = rodar(args.pasta, args.motor, args.limite)
        RESULTADOS.mkdir(parents=True, exist_ok=True)
        df.to_csv(caminho_csv, index=False, encoding="utf-8")

    resumo = resumir(df)
    resumo.to_csv(RESULTADOS / f"{prefixo}_resumo.csv", index=False, encoding="utf-8", float_format="%.4f")

    grafico_cer_por_configuracao(resumo, args.motor, GRAFICOS / f"{prefixo}_cer_por_configuracao.png")
    metadados = args.pasta / "metadados.csv"
    if metadados.exists():
        grafico_cer_vs_inclinacao(df, pd.read_csv(metadados), args.motor,
                                  GRAFICOS / f"{prefixo}_cer_vs_inclinacao.png")
    primeira = args.pasta / df["arquivo"].iloc[0]
    figura_etapas(primeira, EXEMPLOS / "etapas_preprocessamento.png")
    salvar_exemplo(df, args.pasta, EXEMPLOS / f"{prefixo}_exemplo.txt")

    print("\nResumo:")
    print(resumo.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nArquivos salvos em {RESULTADOS}")


if __name__ == "__main__":
    main()
