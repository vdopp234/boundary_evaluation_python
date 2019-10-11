from np_eval_img import eval_img
import numpy as np
import os

epsilon = 1e-7
def join_paths(x, y):
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Empty strings are not acceptable params")
    if x[len(x) - 1] == "/" and y[len(y) - 1] == '/':
        return x[:len(x) - 1] + y
    elif x[len(x) - 1] == "/" or y[len(y) - 1] == '/':
        return x + y
    else:
        return x + '/' + y


def computeRPF(cntR, sumR, cntP, sumP):
    """
    Computes Recall, Precision, and F-Score
    :param cntR:
    :param sumR:
    :param cntP:
    :param sumP:
    :return: elementwise R, P, F
    """
    R = cntR/np.maximum(epsilon, sumR)
    P = cntP/np.maximum(epsilon, sumP)
    F = (2*P*R)/np.maximum(epsilon, P+R)
    return R, P, F


def findBestRPF(T, R, P):
    """
    Linearly Interpolates to find the best threshhold
    :param T: Threshholds
    :param R: Recall Values
    :param P: Precision Values
    :return:
    """
    n_thresholds = T.shape[0]
    if len(T.shape) != 1:
        T = T.flatten()
    if n_thresholds == 1:
        return T, R, P, (2*P*R)/np.maximum(epsilon, P+R)
    A = np.linspace(0, 1, 100)
    B = 1-A
    bstF, bstP, bstR, bstT = -1, -1, -1, -1
    for j in range(1, n_thresholds):
        Rj = R[j]*A + R[j-1]*B
        Pj = P[j]*A + P[j-1]*B
        Tj = T[j]*A + T[j-1]*B
        Fj = 2*Pj*Rj/np.maximum(epsilon, Pj + Rj)
        assert len(Fj.shape) == 1  # Make sure Fj is a row vector
        f, k = np.max(Fj), np.argmax(Fj) # Use F-Score as a metric to get best R, P, T
        k = int(k)  # Sanity check to make sure k is an integer
        if f > bstF:
            bstT = Tj[k]
            bstR = Rj[k]
            bstP = Pj[k]
            bstF = f
    return bstR, bstP, bstF, bstT


def get_metric(model_output_path, ground_truth_path, output_dir=None,
               threshholds=99, max_tolerance=.0075, thin=False):
    """
    Given the path to ONE output and ground truth, return edge evaluation metrics,
    such as ODS, OIS, Average Precision, Recall at 50% Precision
    :param model_output_path: Path to directory containing model outputs
    :param ground_truth_path: Path to directory ground truth boundary maps
    :param output_dir: Path where evaluation output txt files will be written to
    :param threshholds: Number or list of threshholds for evaluation
    :param max_tolerance: Maximum tolerance for edge match
    :param thin: True if thin boundary maps, False otherwise
    :return:
    1) ODS F-Score, Precision, Recall, and Threshhold
    2) OIS F-Score, Precision, and Recall
    3) Average Precision
    4) Recall at 50% Precision
    """
    if not os.path.isdir(model_output_path):
        raise NotADirectoryError("Path to model outputs is not a directory")
    if not os.path.isdir(ground_truth_path):
        raise NotADirectoryError("Path to grouth truth is not a directory")
    if output_dir is None:
        output_dir = join_paths("../data/", "/eval_outputs/")
    model_output_dir = os.listdir(model_output_path)

    n = len(model_output_dir)  # n = number of examples available
    scores = np.zeros((n, 5))
    oisCntR, oisSumR, oisCntP, oisSumP = 0, 0, 0, 0
    cntR = np.zeros((threshholds,))
    cntP = np.zeros((threshholds,))
    sumR = np.zeros((threshholds,))
    sumP = np.zeros((threshholds,))
    T = np.linspace(1/(threshholds+1), threshholds/(threshholds+1), threshholds)
    for i in range(len(model_output_dir)):
        file = model_output_dir[i]
        file_split = file.split('.')
        model_output_file = join_paths(model_output_path, file)
        gt_file = join_paths(ground_truth_path, file_split[0] + '.mat')
        if not os.path.isfile(gt_file):
            raise FileNotFoundError("No matching GT for model prediction: ", gt_file)

        output_file = join_paths(output_dir, file_split[0] + "_eval.txt")
        # if os.path.isfile(output_file):  # If file already exists, skip
        #     print("File exists for evaluation of {}, skipping evaluation".format(file_split[0]))
        #     continue

        # TODO: Look into parallel compute
        # This part may be very slow, performing evaluation on image in directory
        T, cntR1, sumR1, cntP1, sumP1, V = eval_img(model_output_file, gt_file,
                                                    threshholds, max_tolerance,
                                                    thin, save=output_file)
        # print("Completed Initial Evaluation")

        cntR += cntR1
        sumR += sumR1
        cntP += cntP1
        sumP += sumP1

        # Compute OIS (Optimal Image Scaling) Scores

        R1, P1, F1 = computeRPF(cntR1, sumR1, cntP1, sumP1)
        k = np.argmax(F1)
        oisR1, oisP1, oisF1, oisT1 = findBestRPF(T, R1, P1)
        scores[i, :] = [i, oisT1, oisR1, oisP1, oisF1]
        oisCntR += cntR1[k]
        oisCntP += cntP1[k]
        oisSumR += sumR1[k]
        oisSumP += sumP1[k]

    # Compute ODS Scores
    R, P, F = computeRPF(cntR, sumR, cntP, sumP)
    odsR, odsP, odsF, odsT = findBestRPF(T, R, P)
    oisR, oisP, oisF = computeRPF(oisCntR, oisSumR, oisCntP, oisSumP)

    # Compute AP/R50 (Supposedly has a minor bug, according to comment in Matlab source code
    _, k = np.unique(R, return_index=True)
    k = k[::-1]
    R = R[k]
    P = P[k]
    T = T[k]
    F = F[k]
    AP = 0
    if R.size() > 1:
        AP = np.interp(x=R, xp=np.linspace(0, 1, num=100), fp=P)  # May be incorrect
        AP = sum(AP(~np.isnan(AP)))/100
    _, o = np.unique(P, return_index=True)
    R50 = np.interp(P[o], np.maximum(P[o], .5), R[o])

    print("Saving To: ", output_dir)
    np.savetxt(fname=join_paths(output_dir, "/eval_bdry_img.txt"), X=scores)
    np.savetxt(fname=join_paths(output_dir, "/eval_bdry_thr.txt"), X=np.array([T, R, P, F]))
    np.savetxt(fname=join_paths(output_dir, "/eval_bdry.txt"), X=np.array([odsT, odsR, odsP, odsF,
                                                                           AP, oisR, oisP, R50]))


if __name__ == "__main__":
    model_output_path = "../data/prob/test"
    ground_truth_path = "../data/gt/test"
    get_metric(model_output_path=model_output_path, ground_truth_path=ground_truth_path)
