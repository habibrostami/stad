import numpy as np
import matplotlib.pyplot as plt

def color_augment_patches(patch):
    #print(patch)
    low_range = 1.0
    high_range = 1.2
    low_shift = -30
    high_shift = 30
    a_c1 = np.random.uniform(low_range, high_range, 3)
    b_c1 = np.random.uniform(low_shift, high_shift, 3)



    img_patch = np.array(patch, dtype='uint8') # patch * 255

    img_patch_2_r = np.array(np.multiply(img_patch[:, :, 0], a_c1[0]) + b_c1[0], dtype='float')
    img_patch_2_g = np.array(np.multiply(img_patch[:, :, 1], a_c1[1]) + b_c1[1], dtype='float')
    img_patch_2_b = np.array(np.multiply(img_patch[:, :, 2], a_c1[2]) + b_c1[2], dtype='float')

    img_patch_2 = np.dstack((img_patch_2_r, img_patch_2_g, img_patch_2_b))

    # plt.imshow(np.array(scale_range(img_patch_2,0,255),dtype='uint8'))

    augmented_patch = np.array(scale_range(img_patch_2, 0, 255), dtype='uint8') / 255.0
    # img_orig = Image.fromarray(img_patch)
    # img_sample.save('/home/sebastian/Documents/svn/papers/2018/StainingNormalization_Sebastian/images/example_aug_1.png')
    # img_orig.save('/home/sebastian/Documents/svn/papers/2018/StainingNormalization_Sebastian/images/example_orig_2.png')
    #print(augmented_patch)
    return augmented_patch

def scale_range(img, min, max):
    img += -(np.min(img))
    img /= np.max(img) / (max - min + 0.00001)
    img += min
    return img

def to_three_channel_gray_scale(patch):
    #print(patch)
    #a_c = np.random.uniform(0.8, 1.2, 3)
    #b_c = np.random.uniform(-20, 20, 3)
    a_c = 1
    b_c = 0
    img_patch = np.array(patch, dtype='uint8') # patch * 255

    img_patch_2_r = img_patch[:, :, 2]# np.array(np.multiply(img_patch[:, :, 0], a_c[0]) + b_c[0], dtype='float')

    #img_patch_2_r = np.array(np.multiply(img_patch[:, :, 0], a_c[0]) + b_c[0], dtype='float')
    #img_patch_2_g = np.array(np.multiply(img_patch[:, :, 1], a_c[1]) + b_c[1], dtype='float')
    #img_patch_2_b = np.array(np.multiply(img_patch[:, :, 2], a_c[2]) + b_c[2], dtype='float')

    img_patch_2 = np.dstack((img_patch_2_r, img_patch_2_r, img_patch_2_r))

    #plt.imshow(np.array(img_patch_2))

    #augmented_patch = np.array(scale_range(img_patch_2, 0, 255), dtype='uint8') / 255.0
    augmented_patch = np.array(img_patch_2) / 255.0
    # img_orig = Image.fromarray(img_patch)
    # img_sample.save('/home/sebastian/Documents/svn/papers/2018/StainingNormalization_Sebastian/images/example_aug_1.png')
    # img_orig.save('/home/sebastian/Documents/svn/papers/2018/StainingNormalization_Sebastian/images/example_orig_2.png')
    #print(augmented_patch)
    return augmented_patch
