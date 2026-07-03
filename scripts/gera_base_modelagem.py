from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from ml import ATRIBUTOS_MODELO, COLUNA_ALVO, carregar_sinasc, montar_base_modelagem


def main() -> None:
    caminhos = sorted(Path("dados").glob("sinasc_ba_*.parquet"))
    if not caminhos:
        raise FileNotFoundError("Nenhum arquivo sinasc_ba_*.parquet foi encontrado em dados/")

    base_bruta = carregar_sinasc(caminhos)
    base_modelagem = montar_base_modelagem(base_bruta)

    saida = Path("dados/processados/base_modelagem_sinasc.parquet")
    saida.parent.mkdir(parents=True, exist_ok=True)
    base_modelagem.to_parquet(saida, index=False)

    print(f"Arquivo salvo em: {saida}")
    print(f"Linhas: {len(base_modelagem)}")
    print(f"Atributos: {len(ATRIBUTOS_MODELO)}")
    print(f"Alvo: {COLUNA_ALVO}")


if __name__ == "__main__":
    main()
