import re
import string
from collections import Counter


class SquadMetrics:
    """Copied from https://github.com/huggingface/datasets/blob/main/metrics/squad/evaluate.py#L11"""

    @staticmethod
    def f1_score(prediction, ground_truth):
        prediction_tokens = SquadMetrics._normalize_answer(prediction).split()
        ground_truth_tokens = SquadMetrics._normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return SquadMetrics._normalize_answer(prediction) == SquadMetrics._normalize_answer(ground_truth)

    @staticmethod
    def _normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
