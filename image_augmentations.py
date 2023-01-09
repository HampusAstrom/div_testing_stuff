from imgaug import augmenters as iaa
from imgaug.augmenters import Sequential,SomeOf,OneOf,Sometimes,WithColorspace,WithChannels, \
            Noop,Lambda,AssertLambda,AssertShape,Scale,CropAndPad, \
            Pad,Crop,Fliplr,Flipud,Superpixels,ChangeColorspace, PerspectiveTransform, \
            Grayscale,GaussianBlur,AverageBlur,MedianBlur,Convolve, \
            Sharpen,Emboss,EdgeDetect,DirectedEdgeDetect,Add,AddElementwise, \
            AdditiveGaussianNoise,Multiply,MultiplyElementwise,Dropout, \
            CoarseDropout,Invert,ContrastNormalization,Affine,PiecewiseAffine, \
            ElasticTransformation
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imgaug as ia

seq = iaa.Sequential([
    #Sometimes(0.5, PerspectiveTransform(0.05)),
    #Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
    Sometimes(0.5, Affine(scale=(1.0, 1.2))),
    Sometimes(0.5, Affine(rotate=(-180, 180))),
    Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.01) ),
    Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),
    Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
    Sometimes(0.3, Invert(0.2, per_channel=True)),
    Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
    Sometimes(0.5, Multiply((0.6, 1.4))),
    Sometimes(0.5, ContrastNormalization((0.5, 2.2), per_channel=0.3))
    ], random_order=False)

rotate = iaa.Affine(rotate=(-25, 25))

for batch_idx in range(1000):
    # 'image' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale image must have shape (height, width, 1) each.
    # All image must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.

    #image = load_batch(batch_idx)
    image = imageio.imread('data/teapot.png')
    #image = image[:,:,:3]
    image = image.astype(np.uint8)

    print(image.shape)
    #ia.imshow(image)

    image_aug = seq(image=image)
    #image_aug = rotate(image=image)

    print(image_aug.shape)
    ia.imshow(image_aug)

    #img = mpimg.imread('file-name.png')
    #train_on_image(image_aug)
