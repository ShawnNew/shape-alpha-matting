import numpy as np
from config import unknown_code

# compute the MSE error given a prediction, a ground truth and a trimap.
# pred: the predicted alpha matte
# target: the ground truth alpha matte
# trimap: the given trimap
#
def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.
    mask = np.equal(trimap, unknown_code).astype(np.float32)
    loss = np.sum(np.square(error_map) * mask) / np.sum(mask)
    return loss

# compute the SAD error given a prediction, a ground truth and a trimap.
#
def compute_sad_loss(pred, target, trimap):
    error_map = np.abs(pred - target) / 255.
    mask = np.equal(trimap, unknown_code).astype(np.float32)
    loss = np.sum(error_map * mask)

    # the loss is scaled by 1000 due to the large images used in our experiment.
    loss = loss / 1000
    # print('sad_loss: ' + str(loss))
    return loss

def generate_gradient_map(grad, area=3):
    ## Generate gradient map based on computed gradient.
    # This function is used to count the gradient pixels passed a certain small area.
    # Parameters:
    #   grad: a gradient matrix
    #   area: small area to average
    # Output:
    #   grad_map
    num_pixel = area / 2
    col_ = grad.shape[3]
    row_ = grad.shape[2] + 2*num_pixel
    new_row = np.zeros([num_pixel, col_], dtype=np.float32)
    new_col = np.zeros([row_, num_pixel], dtype=np.float32)
    result = np.zeros_like(grad)

    for n in range(grad.shape[0]):
      _tmp = np.r_[new_row, grad[n, 0, :, :], new_row]
      _tmp = np.c_[new_col, _tmp, new_col]
      
      for i in range(result.shape[0]):     # traverse over rows
        for j in range(result.shape[1]):   # traverse over coloumns
          area_count = _tmp[i][j] + _tmp[i][j+1] + _tmp[i][j+2] +\
                       _tmp[i+1][j] + _tmp[i+1][j+1] + _tmp[i+1][j+2] +\
                       _tmp[i+2][j] + _tmp[i+2][j+1] + _tmp[i+2][j+2]
          result[n][0][i][j] = area_count / (area ** 2)
    return result


