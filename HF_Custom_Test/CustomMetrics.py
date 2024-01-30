import numpy as np
import evaluate

class CustomMetrics():
    def compute_metrics(eval_preds):
        accuracy_metrics = evaluate.load('accuracy')
        f1_metrics = evaluate.load('f1')
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        accuracy_score = accuracy_metrics.compute(predictions=predictions, references=labels)
        f1_score = f1_metrics.compute(predictions=predictions, references=labels, average='macro')
        metrics_scores = dict(**accuracy_score, **f1_score)

        return metrics_scores