"""Evaluate exported frame-level probabilities."""

from __future__ import division
import argparse
import csv
import glob
import numpy as np
import os

CSV_SUFFIX = '*.csv'

def import_probs_and_labels(filepath, col_label, col_prob):
    """Import probabilities and labels from csv"""
    filenames = glob.glob(os.path.join(filepath, CSV_SUFFIX))
    assert filenames, "No files found for evaluation"
    labels = []
    probs = []
    for filename in filenames:
        with open(filename) as dest_f:
            for row in csv.reader(dest_f, delimiter=','):
                labels.append(int(row[col_label]))
                probs.append(float(row[col_prob]))
    labels = np.array(labels)
    probs = np.array(probs)

    return probs, labels

def max_search(probs, threshold, mindist):
    """Perform a max search"""
    # Threshold probs
    probabilities = np.copy(probs)
    probabilities[probabilities <= threshold] = 0
    # Potential detections
    idx_p = np.where(probabilities > 0)[0]
    if (idx_p.size == 0):
        return np.zeros(probs.shape)
    # Identify start and end of detections
    p_d = np.diff(idx_p) - 1
    p = np.where(p_d > 0)[0]
    p_start = np.concatenate(([0], p+1))
    p_end = np.concatenate((p, [idx_p.shape[0]-1]))
    # Infer start and end indices of detections
    idx_start = idx_p[p_start]
    idx_end = idx_p[p_end]
    idx_max = [start+np.argmax(probabilities[start:end+1])
        for start, end in zip(idx_start, idx_end)]
    # Remove detections within mindist
    max_diff = np.diff(idx_max)
    carry = 0; rem_i = []
    for i, diff in enumerate(np.concatenate(([mindist], max_diff))):
        if (diff + carry < mindist):
            rem_i.append(i)
            carry += diff
        else:
            carry = 0
    rem_i = np.array(rem_i)
    idx_max_mindist = np.delete(idx_max, rem_i)
    # Return detections
    detections = np.zeros(probabilities.shape, dtype=np.int32)
    detections[idx_max_mindist] = 1
    return detections

def eval_stage_1(probs, labels):
    """Stage 1 evaluation based on frame-level probabilitites"""
    frame_tp_1 = np.intersect1d(np.where(probs >= 0.5), np.where(labels == 1)).shape[0]
    frame_fn_1 = np.intersect1d(np.where(probs < 0.5), np.where(labels == 1)).shape[0]
    frame_rec_1 = frame_tp_1 / (frame_tp_1 + frame_fn_1)
    frame_tp_0 = np.intersect1d(np.where(probs < 0.5), np.where(labels == 0)).shape[0]
    frame_fn_0 = np.intersect1d(np.where(probs >= 0.5), np.where(labels == 0)).shape[0]
    frame_rec_0 = frame_tp_0 / (frame_tp_0 + frame_fn_0)
    uar = (frame_rec_1 + frame_rec_0) / 2.0
    return uar

def eval_stage_2(dets, labels):
    """Stage 2 evaluation based on gesture-level metric proposed by Kyritsis et al. (2019)"""
    def _split_idx(labels):
        idx_t = np.where(labels == 1)[0]
        t_d = np.diff(idx_t) - 1
        t = np.where(t_d > 0)[0]
        t_start = np.concatenate(([0], t+1))
        t_end = np.concatenate((t, [idx_t.shape[0]-1]))
        idx_start = idx_t[t_start]
        idx_end = idx_t[t_end]
        return [np.arange(start, end+1) for start, end in zip(idx_start, idx_end)]
    idxs_t = _split_idx(labels)
    idxs_f = np.where(labels == 0)
    splits_t = [dets[split_idx] for split_idx in idxs_t]
    splits_f = dets[idxs_f]
    tp = np.sum([1 if np.sum(split) > 0 else 0 for split in splits_t])
    fn = np.sum([0 if np.sum(split) > 0 else 1 for split in splits_t])
    fp_1 = np.sum([np.sum(split)-1 if np.sum(split)>1 else 0 for split in splits_t])
    fp_2 = np.sum(splits_f)
    if tp > 0:
        prec = tp / (tp + fp_1 + fp_2)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
    else:
        prec = 0
        rec = 0
        f1 = 0
    return tp, fn, fp_1, fp_2, prec, rec, f1

def main(args=None):
    # Import the probs and labels from csv
    probs, labels = import_probs_and_labels(args.prob_dir, args.col_label, args.col_prob)
    # Calculate UAR for Stage I
    uar = eval_stage_1(probs, labels)
    print('UAR: {}'.format(uar))
    # Perform grid search
    if args.mode == 'estimate':
        # All evaluated threshold values
        threshold_vals = np.arange(args.min_threshold, args.max_threshold, args.inc_threshold)
        f1_results = []
        for threshold in threshold_vals:
            # Perform max search
            dets = max_search(probs, threshold, args.min_dist)
            # Calculate Stage II
            _, _, _, _, _, _, f1 = eval_stage_2(dets, labels)
            f1_results.append(f1)
        # Find best threshold
        final_threshold = threshold_vals[np.argmax(f1_results)]
        final_dets = max_search(probs, final_threshold, args.min_dist)
        tp, fn, fp_1, fp_2, prec, rec, f1 = eval_stage_2(final_dets, labels)
        print('-----')
        print('Best threshold: {}'.format(final_threshold))
        print('-----')
        print('F1: {}'.format(f1))
        print('Precision: {}'.format(prec))
        print('Recall: {}'.format(rec))
        print('-----')
        print('TP: {}'.format(tp))
        print('FN: {}'.format(fn))
        print('FP_1: {}'.format(fp_1))
        print('FP_2: {}'.format(fp_2))
    else:
        # Perform max search
        dets = max_search(probs, args.threshold, args.min_dist)
        # Calculate Stage II
        tp, fn, fp_1, fp_2, prec, rec, f1 = eval_stage_2(dets, labels)
        print('-----')
        print('F1: {}'.format(f1))
        print('Precision: {}'.format(prec))
        print('Recall: {}'.format(rec))
        print('-----')
        print('TP: {}'.format(tp))
        print('FN: {}'.format(fn))
        print('FP_1: {}'.format(fp_1))
        print('FP_2: {}'.format(fp_2))

# Run
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model Stage II')
    parser.add_argument('--prob_dir', type=str, default='eval', nargs='?', help='Directory with eval data.')
    parser.add_argument('--min_dist', type=int, default=128, nargs='?', help='Minimum frames between detections.')
    parser.add_argument('--threshold', type=float, default=0.9, nargs='?', help='Detection threshold probability')
    parser.add_argument('--mode', type=str, default='evaluate', nargs='?', help='Evaluation or estimation and evaluation')
    parser.add_argument('--min_threshold', type=float, default=0.5, nargs='?', help='Minimum detection threshold probability')
    parser.add_argument('--max_threshold', type=float, default=1, nargs='?', help='Maximum detection threshold probability')
    parser.add_argument('--inc_threshold', type=float, default=0.001, nargs='?', help='Increment for detection threshold search')
    parser.add_argument('--col_label', type=int, default=1, nargs='?', help='Col number of label in csv')
    parser.add_argument('--col_prob', type=int, default=2, nargs='?', help='Col number of probability in csv')
    args = parser.parse_args()
    main(args)
