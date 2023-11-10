import os
from matplotlib import pyplot as plt
from pydantic import Field

from endaaman.ml import BaseMLCLI
from .fold import FoldDataset

J = os.path.join


class CLI(BaseMLCLI):
    class InspectArgs(BaseMLCLI.CommonArgs):
        source: str = 'cache/enda2_512/'
        code: str = 'LMGGG'
        fold: int = 0
        total_fold: int = 6
        show: bool = Field(False, cli=('--show', ))
        limit: int = -1

    def run_inspect(self, a):
        ds = FoldDataset(
            fold=a.fold,
            total_fold=a.total_fold,
            code=a.code,
            source_dir=a.source,
            limit=a.limit,
            target='all',
        )
        fig = ds.inspect()
        # plt.savefig(J(a.source, 'balance{a.fold}_{a.code}.png'))
        if a.show:
            plt.show()

    def run_i(self, a):
        self.ds = FoldDataset(
            fold=0,
            total_fold=6,
            code='LMGGG',
            source_dir='cache/enda2_512',
            minimum_area=0.7,
            limit=10,
            target='all',
        )

if __name__ == '__main__':
    print('aa')
    cli = CLI()
    cli.run()
