from keras.callbacks import Callback

accuracy_threshold = 98e-2


class StopByAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= accuracy_threshold:
            print('Accuracy has reach = %2.2f%%' % (logs['accuracy'] * 100), 'training has been stopped.')
            self.model.stop_training = True
