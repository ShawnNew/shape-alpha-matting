import evaluation
from evaluation import alphaNetModel
from PIL import Image
import cv2
from scipy import misc
import numpy as np
import os
import glog as log
from utils import compute_mse_loss
from config import _model, _weights, source
from config import net_input_w, net_input_h
from config import unknown_code

if __name__ == "__main__":
    _model = alphaNetModel(_model, _weights, "gpu", 2)
    _mse = 0
    time_ = 0
    with open(source, 'r') as f:
        lines_ = f.readlines()
        nums = len(lines_)
        log.info("Starting processing.")
        for i, line_ in enumerate(lines_):
            base, _ = os.path.split(source)
            base += "/"
            items = line_.rstrip().replace("./", base).split(" ")
            item_name = os.path.basename(items[0])
            # read images from datasets directory
            data = misc.imresize(
                    np.asarray(cv2.imread(items[6])), \
                    [net_input_w, net_input_h], \
                    interp='nearest').astype(np.float64)
            tri_map_original = np.asarray(Image.open(items[1]))
            tri_map = misc.imresize(
                    tri_map_original, \
                    [net_input_w, net_input_h], \
                    interp='nearest').astype(np.float64)
            gt = np.asarray(Image.open(items[4])).astype(np.float64)
            tri_map = np.expand_dims(tri_map, axis=2)

            original_shape = gt.shape

            # testing
            feed_data_ = np.concatenate(
                [data, tri_map], axis=2)
            feed_data_ = np.expand_dims(
                np.transpose(feed_data_, (2, 0, 1)), axis=0)
            _model.feed_input(feed_data_)
            duration, pred = _model.predict_without_shape_data()
            log.info("Processed %s, consumed %f second."% (item_name, duration))
            pred = np.where(np.equal(tri_map[:, :, 0], unknown_code), pred, tri_map[:, :, 0])
            pred = cv2.resize(
                pred, (original_shape[1], original_shape[0]), \
                interpolation=cv2.INTER_CUBIC
            )
            mse = compute_mse_loss(pred, gt, tri_map_original)
            _mse += mse
            log.info("mse for %s is: %f"% (item_name, mse))
            output_img = Image.fromarray(pred)
            output_dir = os.path.join("../", "test-output")
            if not os.path.exists(output_dir): os.mkdir(output_dir)
            output_filename = os.path.join(output_dir, item_name)
            output_img.save(output_filename)
            time_ += duration

        _mse /= nums
        log.info("Mean time consumption every single image is: %f"% (time_ / nums))
        log.info("Mean mse is: %f"% _mse)
