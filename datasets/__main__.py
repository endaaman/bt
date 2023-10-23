from matplotlib import pyplot as plt
from pydantic import Field

from endaaman.ml import BaseMLCLI
from .fold import FoldDataset


class CLI(BaseMLCLI):
    class InspectArgs(BaseMLCLI.CommonArgs):
        source: str
        code: str
        fold: int
        show: bool = Field(False, cli=('--show', ))

    def run_inspect(self, a):
        ds = FoldDataset(
            fold=0,
            total_fold=a.fold,
            code=a.code,
            source_dir=f'cache/images/{a.source}',
            target='all',
            )
        fig = ds.inspect()
        plt.savefig(f'cache/images/enda2_512/balance{a.fold}_{a.code}.png')
        if a.show:
            plt.show()

if __name__ == '__main__':
    cli = CLI()
    cli.run()
