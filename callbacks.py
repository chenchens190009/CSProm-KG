import time
from pytorch_lightning.callbacks import Callback


class PrintingCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.start = time.time()
        print('Epoch: %4d, ' % trainer.current_epoch, flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if len(pl_module.history['loss']) == 0:
            return
        loss = pl_module.history['loss']
        avg_loss = sum(loss) / len(loss)
        pl_module.history['loss'] = []
        print('Total time: %4ds, loss: %2.4f' % (int(time.time() - self.start), avg_loss), flush=True)
        print('-' * 50, flush=True)

    def on_validation_start(self, trainer, pl_module):
        if hasattr(self, 'start'):
            print('Training time: %4ds' % (int(time.time() - self.start)), flush=True)
        self.val_start = time.time()

    def on_validation_end(self, trainer, pl_module):
        print(pl_module.history['perf'], flush=True)
        print('Validation time: %4ds' % (int(time.time() - self.val_start)), flush=True)

    def on_test_end(self, trainer, pl_module):
        print(flush=True)
        print('=' * 50, flush=True)
        print('Epoch: test', flush=True)
        print(pl_module.history['perf'], flush=True)
        print('=' * 50, flush=True)