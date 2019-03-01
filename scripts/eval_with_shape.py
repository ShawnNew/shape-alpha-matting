import evaluation
from evaluation import ShapeAlphaNetModel
from PIL import Image
import cv2
from scipy import misc
import numpy as np
import os
import glog as log
from utils import compute_mse_loss
from config import shape_model, shape_weights, source
from config import net_input_w, net_input_h
from config import unknown_code
import pdb

if __name__ == "__main__":
    shape_model = ShapeAlphaNetModel(shape_model, shape_weights, "gpu", 2)
    #shape_mse = 0
    raw_mse = 0
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
            tri_map_enlarge = np.where(np.equal(tri_map_original, 255), 128, tri_map_original)
            tri_map = misc.imresize(
                    tri_map_enlarge, \
                    [net_input_w, net_input_h], \
                    interp='nearest').astype(np.float64)
            gradient = misc.imresize(
                    np.asarray(Image.open(items[2])), \
                    [net_input_w, net_input_h], \
                    interp='nearest').astype(np.float64)
            gt = np.asarray(Image.open(items[4])).astype(np.float64)
            tri_map = np.expand_dims(tri_map, axis=2)
            gradient = np.expand_dims(gradient, axis=2)

            original_shape = gt.shape

            # testing
            feed_data_ = np.concatenate(
                [data, tri_map, gradient], axis=2)#, roughness], axis=2)
            feed_data_ = np.expand_dims(
                np.transpose(feed_data_, (2, 0, 1)), axis=0)
            shape_model.feed_input_with_shape(feed_data_)
            duration, raw_output = shape_model.predict_with_shape_data()
            log.info("Processed %s, consumed %f second."% (item_name, duration))
            #pred = np.where(np.equal(tri_map[:, :, 0], unknown_code), pred, tri_map[:, :, 0])
            #pred = cv2.resize(
            #    pred, (original_shape[1], original_shape[0]), \
            #    interpolation=cv2.INTER_CUBIC
            #)
            raw_output = np.where(
                                 np.equal(tri_map[:,:,0], unknown_code),\
                                 raw_output, tri_map[:,:,0])
            raw_output = cv2.resize(
                                   raw_output, (original_shape[1], original_shape[0]), \
                                   interpolation=cv2.INTER_CUBIC)
            #mse = compute_mse_loss(pred, gt, tri_map_enlarge)
            #shape_mse += mse
            mse = compute_mse_loss(raw_output, gt, tri_map_enlarge)
            raw_mse += mse
            log.info("mse for %s is: %f"% (item_name, mse))
            #output_img = Image.fromarray(pred).convert("L")
            #output_dir = os.path.join("../", "shape-test-output")
            raw_output_img = Image.fromarray(raw_output).convert("L")
            raw_output_dir = os.path.join("../", "shape-test-output")
            #if not os.path.exists(output_dir): os.mkdir(output_dir)
            if not os.path.exists(raw_output_dir): os.mkdir(raw_output_dir)
            #output_filename = os.path.join(output_dir, item_name)
            #output_img.save(output_filename)
            raw_output_filename = os.path.join(raw_output_dir, item_name)
            raw_output_img.save(raw_output_filename)
            time_ += duration

        #shape_mse /= nums
        raw_mse /= nums
        log.info("Mean time consumption every single image is: %f"% (time_ / nums))
        log.info("Mean mse of raw alpha is: %f"% raw_mse)
        #log.info("Mean mse of refiner is: %f"% shape_mse)
            
