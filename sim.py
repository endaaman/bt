class TrainerConfig(BaseTrainerConfig):
    model_name:str
    source: str = 'enda4_512'
    total_fold: int = 5
    fold: int = 0
    size: int = 256
    minimum_area: float = 0.6
    limit: int = 100
    noupsample: bool = False
    scheduler: str = 'plateau_10'

    mean: float = MEAN
    std: float = STD

    def num_classes(self):
        return len(self.unique_code())


class Trainer(BaseTrainer):
    def prepare(self):
        model = SimSiamModel()
        return model

    def create_optimizer(self):
        return optim.RAdam(params)

    def eval(self, inputs, gts, i):
        inputs = inputs.to(self.device)
        gts = gts.to(self.device)
        logits = self.model(inputs, activate=False)
        preds = torch.softmax(logits, dim=1)
        loss = self.criterion(logits, gts)
        if self.config.nested == 'graph':
            graph_loss = self.model.graph_matrix(preds, gts)
            loss = loss + graph_loss
            if i%500 == 0:
                torch.set_printoptions(precision=2, sci_mode=False)
                print(self.model.graph_matrix.matrix)
                m = self.model.graph_matrix.get_matrix()
                print(m.softmax(dim=0))
        elif self.config.nested == 'hier':
            h_loss = self.model.hier_matrixes(preds, gts)
            loss = loss + h_loss
            if i%500 == 0:
                torch.set_printoptions(precision=2, sci_mode=False)
                print(self.model.hier_matrixes.matrixes[0].softmax(dim=1))
                print(self.model.hier_matrixes.matrixes[1].softmax(dim=1))
                print(self.model.hier_matrixes.matrixes[2].softmax(dim=1))
                print(self.model.hier_matrixes.matrixes[3].softmax(dim=1))
        return loss, logits.detach().cpu()

    def _visualize_confusion(self, ax, label, preds, gts):
        preds = torch.argmax(preds, dim=-1)
        gts = gts.flatten()
        cm = skmetrics.confusion_matrix(gts.numpy(), preds.numpy())
        ticks = [*self.train_dataset.unique_code]
        sns.heatmap(cm, annot=True, ax=ax, fmt='g', xticklabels=ticks, yticklabels=ticks)
        ax.set_title(label)
        ax.set_xlabel('Predict', fontsize=13)
        ax.set_ylabel('GT', fontsize=13)

    def visualize_train_confusion(self, ax, train_preds, train_gts, val_preds, val_gts):
        self._visualize_confusion(ax, 'train', train_preds, train_gts)

    def visualize_val_confusion(self, ax, train_preds, train_gts, val_preds, val_gts):
        if val_preds is None or val_gts is None:
            return
        self._visualize_confusion(ax, 'val', val_preds, val_gts)

    def create_scheduler(self):
        s = self.config.scheduler
        m = re.match(r'^plateau_(\d+)$', s)
        if m:
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=int(m[1]))
        m = re.match(r'^step_(\d+)$', s)
        if m:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=int(m[1]), gamma=0.1)
        raise RuntimeError('Invalid scheduler')
        # return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1)

    def continues(self):
        lr = self.get_current_lr()
        return lr > 1e-7

    def metrics_precision(self, preds, gts, batch):
        if batch:
            return None
        preds = preds.detach().cpu()
        labels = torch.unique(preds)
        correct = 0
        for label in labels:
            items = gts[preds == label]
            elements, counts = torch.unique(items, return_counts=True)
            dominant = elements[torch.argmax(counts)]
            # print(label, dominant, torch.sum(items == dominant))
            correct += torch.sum(items == dominant)
        return correct/len(preds)



class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class TrainArgs(BaseTrainArgs):
        source: str = 'enda4_512'
        model_name: str = Field('vit', l='--model', s='-m')
        batch_size: int = Field(16, s='-B')
        code: str = 'LMGGG_'
        base_lr: float = 1e-5 # ViT for 1e-6
        lr: float = -1
        total_fold: int = 5
        fold: int = 0
        minimum_area: float = 0.6
        limit: int = 500
        noupsample: bool = False
        num_classes: int = Field(3, s='-C')

        num_workers: int = Field(4, s='-N')
        epoch: int = Field(100, s='-E')
        suffix: str = Field('', s='-S')
        prefix: str = ''
        overwrite: bool = Field(False, s='-O')

    def run_train(self, a:TrainArgs):
        m = re.match(r'^.*_(\d+)$', a.source)
        assert m
        size = int(m[1])
        lr = a.lr if a.lr>0 else a.base_lr*a.batch_size

        config = TrainerConfig(
            source = a.source,
            model_name = a.model_name,
            batch_size = a.batch_size,
            size = size,
            lr = lr,
            code = a.code,
            num_classes = a.num_classes,
            total_fold = a.total_fold,
            fold = a.fold,
            minimum_area = a.minimum_area,
            limit = a.limit,
            upsample = not a.noupsample,
        )

        if a.fold < 0:
            dss = [DinoFoldDataset(
                 source_dir = J('data/tiles', a.source),
                 total_fold = a.total_fold,
                 fold = -1,
                 target = 'all',
                 code = a.code,
                 size = size,
                 minimum_area = a.minimum_area,
                 limit = a.limit,
                 upsample = config.upsample,
                 augmentation = True,
                 normalization = True,
            ), None]
        else:
            dss = [
                DinoFoldDataset(
                    source_dir = J('data/tiles', a.source),
                    total_fold = a.total_fold,
                    fold = a.fold,
                    target = t,
                    code = a.code,
                    size = size,
                    minimum_area = a.minimum_area,
                    limit = a.limit,
                    upsample = config.upsample and t=='train',
                    augmentation= t=='train',
                    normalization = True,
                ) for t in ('train', 'test')
            ]

        out_dir = J(
            'out', f'{a.source}_dino', config.code,
            f'fold{a.total_fold}_{a.fold}', a.prefix, config.model_name,
        )
        if a.suffix:
            out_dir += f'_{a.suffix}'

        trainer = Trainer(
            config=config,
            out_dir=out_dir,
            train_dataset=dss[0],
            val_dataset=dss[1],
            use_gpu=not a.cpu,
            overwrite=a.overwrite,
            experiment_name=a.source,
        )

        trainer.start(a.epoch)

