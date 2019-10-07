import numpy as np
from EdgeEval import EdgeEval
from scipy import io, misc
import matplotlib.pyplot as plt
import os, sys
from skimage.morphology import thin
from collections import Counter
from tqdm import tqdm
from imageio import imread, imsave
import time


def getPRGT(img_id, path):
    """
    Returns matlab-calculated PR values
    """
    with open(os.path.join(path, '%d_ev1.txt'%imgId), 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]
        thresh, cntR, sumR, cntP, sumP = [np.zeros(len(lines)).astype(np.double) for _ in range(5)]
        for lid, line in enumerate(lines):
            thresh[lid], cntR[lid], sumR[lid], cntP[lid], sumP[lid] = [float(num) for num in line]
    return thresh, cntR, sumR, cntP, sumP

def boundaryPR(pb, segs, nthresh=100.):
    """
    Returns cythonized matlab values
    """
    edge_eval = EdgeEval()

    thresh = np.linspace(1./nthresh, 1-1./nthresh, nthresh-1).astype(np.double)

    cntR = np.zeros_like(thresh)
    sumR = np.zeros_like(thresh)
    cntP = np.zeros_like(thresh)
    sumP = np.zeros_like(thresh)

    for t in range(len(thresh)):
        # threshold pb to get binary boundary map
        bmap = (pb >= thresh[t])
        bmap = thin(bmap)
        bmap = bmap.copy(order='C')  # the Fortran order is necessary for code to work correctly
                                    # VD: Getting segfault if order is not C    
        bmap = bmap.astype(np.double)
        # accumulate machine matches, since the machine pixels are
          # allowed to match with any segmentation
        accP = np.zeros_like(pb).astype(np.bool_)
        for sid, seg in enumerate(segs):
            # compute the correspondence
            # init = time.time()
            match1 = edge_eval.getOut1(bmap, seg)
            # first = time.time()
            match2 = edge_eval.getOut2(bmap, seg)
            # second = time.time()
            # print("First Correspondence: {}\n Second Correspondence: {}".format(first - init, second - first))
            # accumulate machine matches
            accP = accP | (match1 > 0)
            # compute recall
            sumR[t] += seg.sum()
            cntR[t] += (match2 > 0).sum()
        # compute precision
        sumP[t] += bmap.sum()
        cntP[t] += accP.sum()

        # print(thresh[t], cntR[t], sumR[t], cntP[t], sumP[t])
    return thresh, cntR, sumR, cntP, sumP


if __name__ == '__main__':

    datapath = '../data'
    gtpath = os.path.join(datapath, 'gt/test')
    predpath = os.path.join(datapath, 'nms/test')
    evalpath = os.path.join(datapath, 'eval')
    resultpath = os.path.join(datapath, 'results')
    os.makedirs(evalpath, exist_ok=True)
    os.makedirs(resultpath, exist_ok=True)
    
    ids = sorted([int(fn[:-4]) for fn in os.listdir(predpath) if 'png' in fn])
    for imgId in tqdm(ids):
        print("Processing img %d" % imgId)
        pb = imread(os.path.join(predpath, '%d.png'%imgId)).astype(np.double)
        pb /= 255.  # normalize value to [0,1]

        gt = io.loadmat(os.path.join(gtpath, '%d.mat'%imgId))['groundTruth'][0]
        segs = []
        for segment in range(gt.shape[0]):
            seg = gt[segment][0][0][1]
            seg = seg.astype(np.double).copy(order='C')
            segs.append(seg)

        thresh_gt, cntR_gt, sumR_gt, cntP_gt, sumP_gt = getPRGT(imgId, evalpath)
        R_gt = cntR_gt / (sumR_gt + (sumR_gt==0))
        P_gt = cntP_gt / (sumP_gt + (sumP_gt==0))
        print("Matlab: P: {}, R: {}".format(P_gt, R_gt))
        # plt.plot(R_gt, P_gt, 'b', label='MATLAB', linewidth=4)

        start = time.time()
        thresh, cntR, sumR, cntP, sumP = boundaryPR(pb, segs)
        finish = time.time()
        print("Total Op Speed: ", finish - start)
        eps = 1e-5
        R = cntR / np.maximum(eps, sumR)
        P = cntP / np.maximum(sumP, eps)
        # print("Python: P: {}, R: {}".format(P, R))
        # np.savetxt(fname="py_eval_results/{}.txt".format(imgId), X=np.array([thresh, cntR, sumR, cntP, sumP]))
        # print("Saved {} Eval Result".format(imgId))
        # print("Absolute Difference Precision: ", np.abs(P_gt - P))
        # print("Absolute Difference Recall: ", np.abs(R_gt - R))
        # print("Precision L1 Error: ", np.linalg.norm((P_gt - P).flatten(), ord=1))
        # print("Recall L1 Error: ", np.linalg.norm((R_gt - R).flatten(), ord=1))

        # plt.plot(R, P, 'r', label='Python')
        # plt.legend()
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.savefig(os.path.join(resultpath, '%d.png'%imgId))
        # plt.show()
        # plt.close()
