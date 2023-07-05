from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def __init__(self):
            self.is_first_log = False

    def on_train_begin(self, args, state, control, logs=None, **kwargs):
        print('\n학습을 시작합니다.\n')

    def on_train_end(self, args, state, control, logs=None, **kwargs):
        print('\n학습이 끝났습니다.\n')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            out_logs = {}
            if 'loss' in logs:
                out_logs['loss'] = f"{logs['loss']:.6}"
            if 'learning_rate' in logs:
                out_logs['learning_rate'] = f"{logs['learning_rate']:.6}"
            if 'eval_loss' in logs:
                out_logs['eval_loss'] = f"{logs['eval_loss']:.6}"
            if 'eval_accuracy' in logs:
                out_logs['eval_accuracy'] = f"{logs['eval_accuracy']:.6}"
            if 'eval_f1' in logs:
                out_logs['eval_f1'] = f"{logs['eval_f1']:.6}"
            if out_logs and 'epoch' in logs:
                out_logs['epoch'] = f"{logs['epoch']:.6}"
            if out_logs:
                output = ' '.join([f'{key}: {str(value).ljust(10)}' for key, value in out_logs.items()])
                print(output)
