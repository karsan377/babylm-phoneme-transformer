import numpy as np

from scripts.train import WordTracker


def test_word_tracker_across_epochs():
    tracker = WordTracker(np.asarray([10, 20, 30, 40]), effective_batch_size=2)
    assert tracker.steps_per_epoch == 2
    assert tracker.words_at_step(1) == 30
    assert tracker.words_at_step(2) == 100
    assert tracker.words_at_step(3) == 130
