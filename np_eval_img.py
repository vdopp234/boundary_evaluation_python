import numpy as np
import cv2
from skimage import morphology
import scipy.io


def join_paths(x, y):
    if x[len(x) - 1] == "/" and y[len(y) - 1] == '/':
        return x[:len(x) - 1] + y
    elif x[len(x) - 1] == "/" or y[len(y) - 1] == '/':
        return x + y
    else:
        return x + '/' + y


def load_img(path):
    assert type(path) == str
    file_type = path[len(path)-4:]
    if file_type not in ['.jpg', '.jpeg', '.png']:
        raise ValueError("Incompatible file type")
    return cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE).astype(np.float64)/255 # Normalizes model output


def load_mat(path, key='groundTruth'):
    gt_mat = scipy.io.loadmat(path)[key][0]
    n = gt_mat.shape[0]
    lst = []
    for i in range(n):
        holder_mat = gt_mat[i][0][0][1]  # Get rid of wrapper arrays, access boundaries key
        lst.append(holder_mat.astype(np.float64))
    return np.concatenate([np.expand_dims(x, axis=0) for x in lst], axis=0)


def correspond_pixels(output_map, gt_map, max_tolerance=.01, outlier_cost=100):
    """
    Implements correspondPixels function in Matlab
    :param output_map:
    :param gt_map:
    :return:
    """
    
    raise NotImplementedError


def eval_img(model_output_path, ground_truth_path, threshhold=99,
             max_tolerance=.0075, thin=True, save=None):
    """
    Helper function, given image returns precision, recall, and f-score of boundary prediction
    :param model_output_path: Path to output of model
    :param ground_truth_path: Path to ground truth
    :param threshhold: An int,
    :param max_tolerance: Equivalent to epsilon for np.isclose(), returns distance
    :param thin: Whether we want to thin boundary. See skimage.morphology for more details
    :param save: If None, don't save the intermediate result. Else, save to path specified at save
    :return:
    """
    epsilon = 1e-10
    K = threshhold
    threshhold = np.linspace(start=1/(K+1), stop=K/(K+1), num=K)

    model_output_img = load_img(model_output_path)
    ground_truth_img = load_mat(ground_truth_path)
    n = ground_truth_img.shape[0]

    cntR = np.zeros_like(threshhold)
    sumR = np.zeros_like(threshhold)
    cntP = np.zeros_like(threshhold)
    sumP = np.zeros_like(threshhold)
    v = np.zeros(list(model_output_img.shape) + [3, K])

    for k in range(K):  # Iterating through thresholds
        model_output_img1 = (model_output_img >= max(epsilon, threshhold[k]))
        if thin:  # Toggled if the output is a segmentation?
            model_output_img1 = morphology.thin(model_output_img1)  # Thins the model output, see https://www.mathworks.com/help/images/ref/bwmorph.html
        match_pred = np.zeros_like(model_output_img)
        match_gt = np.zeros_like(model_output_img)
        all_gt = np.zeros_like(model_output_img)
        for g in range(n):
            match_pred1, match_gt1 = correspond_pixels(model_output_img1, ground_truth_img[g], max_tolerance=max_tolerance)
            match_pred = np.logical_or(match_pred, (match_pred1>0).astype(np.float64)).astype(np.float64)
            match_gt = match_gt + (match_gt > 0).astype(np.float64)
            all_gt = all_gt + ground_truth_img[g]

        cntR[k] = np.sum(match_gt)
        sumR[k] = np.sum(all_gt)
        cntP[k] = np.sum(np.isclose(match_pred, np.zeros_like(match_pred)))  # Counts number of nonzero elements
        sumP[k] = np.sum(np.isclose(model_output_img1, np.zeros_like(model_output_img1)))

        cs = np.array([
            [1, 0, 0],
            [0, .7, 0],
            [.7, .8, 1]
        ])
        cs = cs - 1
        FP = model_output_img1 - match_pred
        TP = match_pred
        FN = (all_gt - match_gt)/n
        for g in range(3):
            v[:, :, g, k] = np.maximum(np.zeros_like(FN), 1+FN*cs[0, g] + TP*cs[1, g] + FP*cs[2, g])

        x = v.shape[1] - 1
        y = v.shape[0] - 1
        # print(v.shape)
        # print(v[: 1:, :, k].shape)
        # print(v[:, :x, :, k].shape)
        v[:, 1:, :, k] = np.minimum(v[:, 1:, :, k], v[:, :x, :, k])
        v[1:, :, :, k] = np.minimum(v[1:, :, :, k], v[:y, :, :, k])
        if save:
            # Ignore this error, code is correct
            np.savetxt(fname=save, X=np.array([threshhold, cntR, sumR, cntP, sumP]))
        print(threshhold)
    return threshhold, cntR, sumR, cntP, sumP, v


if __name__ == "__main__":
    model_output_path_ = "/home/vishnu234/Desktop/Interactive_Segmentation/data/prob/test/2018.png"
    ground_truth_path_ = "/home/vishnu234/Desktop/Interactive_Segmentation/data/gt/test/2018.mat"
    print(eval_img(model_output_path_, ground_truth_path_, thin=False))

