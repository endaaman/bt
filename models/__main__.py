from . import *
from .loss import *


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    def run_loss(self, a:CommonArgs):
        n = NestedCrossEntropyLoss(
            rules=[
                {
                    'weight': 1000.0, 'index': [],
                },
                {
                    'weight': 1.0, 'index': [[2, 3, 4]],
                }
            ])

        # loss should be different
        x0 = torch.tensor([[1., 0, 0, 3, 0], [1., 0, 0, 3, 0]])
        y0 = torch.tensor([2, 2])

        # loss should be same
        x1 = torch.tensor([[1., 0, 0, 0, 3], [1., 0, 0, 0, 3]])
        y1 = torch.tensor([1, 1])

        print('nested')
        print('x0 y0')
        print( n(x0, y0) )
        print('x1 y1')
        print( n(x1, y1) )

        c = CrossEntropyLoss()
        print()
        print('normal')
        print('x0 y0')
        print( c(x0, y0) )
        print('x1 y1')
        print( c(x1, y1) )

    class ModelArgs(CommonArgs):
        name : str = Field('tf_efficientnetv2_b0', s='-n')

    def run_model(self, a):
        # model = AttentionModel(name=a.model, num_classes=3)
        # y, aa = model(x, with_attentions=True)

        model = IICModel(name=a.name, num_classes=3, num_classes_over=10)
        x = torch.rand([4, 3, 512, 512])
        y, y_over, f = model(x, with_feautres=True)
        print('y', y.shape)
        print('y_over', y_over.shape)
        print('f', f.shape)

    def run_mat(self, a):
        initial_value = torch.tensor([
            [10, .0, .0, .0, .0, ],
            [.0, 10, .0, .0, .0, ],
            [.0, .0, 5, 10, .0, ],
            [.0, .0, .0, 5, .0, ],
            [.0, .0, .0, .0, 10, ],
        ]).float().clamp(1e-16)
        g = GraphMatrix(5, initial_value)
        # preds = torch.randn([3, 5]).softmax(dim=1)
        preds = torch.tensor([
            [.6, .1, .1, .1, .1],
            [.1, .6, .1, .1, .1],
            [.1, .1, .6, .1, .1],
            [.1, .1, .1, .6, .1],
            [.1, .1, .1, .1, .6],
        ])
        gts = torch.tensor([2, 2, 2, 2, 2])
        print(g(preds, gts, by_index=True))

    def run_hier(self, a):
        model = TimmModel('resnet18', 6)
        hier_matrixes = HierMatrixes(6)
        h = TimmModelWithHier(model, hier_matrixes)

        ii = torch.randn(4, 3, 256, 256)
        ll = h(ii)

        gts = torch.tensor([0, 1, 2, 3])

        loss = hier_matrixes(ll, gts)
        print(ll.shape)
        print(loss)


    def run_sim(self, a):
        model = SimSiamModel('resnet18', num_neck=512, num_features=2048)

        t = torch.randn(4, 2, 3, 256, 256)

        z0, z1, p0, p1 = model(t)

        c = SymmetricCosSimLoss()
        loss = c(z0.detach(), z1.detach(), p0, p1)

        print(z0.shape)
        print(z1.shape)
        print(p0.shape)
        print(p1.shape)
        print(loss)

    def run_barlow(self, a):
        model = BarlowTwinsModel('resnet18', num_features=2048)

        x0 = torch.randn(2, 3, 256, 256)
        x1 = torch.randn(2, 3, 256, 256)

        print(x0)

        z0 = model(x0)
        z1 = model(x1)
        print(z0)
        print(z1)

        c = BarlowTwinsLoss()
        loss = c(z0, z1)
        print(loss)

    def run_ctranspath(self, a):
        m = CompareModel(6, base='ctranspath')
        t = torch.rand(4, 3, 224 ,224)
        print(m(t).shape)



if __name__ == '__main__':
    cli = CLI()
    cli.run()
