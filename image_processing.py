import io
import PIL
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from skimage import color, exposure, morphology, measure, segmentation

class Sketch2Model:
    def __init__(self, buf):
        self.initial_image = load_image(buf)
        self.process()
        
    def process(self):
        """Sketch2Model uses morphological filtering to label regions in a
        hollow sketch and assign them discrete labels"""
        
        ## Compressor
        im = color.rgb2gray(self.initial_image[0:-1:2,0:-1:2])
        flt2 = sp.ndimage.filters.gaussian_filter(im**2, 21)
        self.compressed = im/(np.sqrt(flt2))

        ## Contrast streching
        p2, p98 = np.percentile(self.compressed, (2, 98))
        self.contrasted = exposure.rescale_intensity(self.compressed, in_range=(p2, p98))

        ## Binarization with scalar threshold           
        self.binary = ~(color.rgb2gray(self.contrasted) > 0.5)

        ## Use binary closing to connect lines that aren't
        ## completely crossing
        self.closed = morphology.binary_closing(self.binary)

        ## Remove small objects (bright and dark spots)
        removed = morphology.remove_small_objects(self.closed)
        self.removed = ~morphology.remove_small_objects(~removed)

        ## Skeletonize and dilate to get final boundaries
        edges = morphology.skeletonize(self.removed)
        self.skeletonized = ~morphology.dilation(edges, morphology.disk(1))

        ## Label regions
        clean_border = segmentation.clear_border(self.skeletonized)
        self.labeled = measure.label(clean_border, background = 999999) + 1

        ## Change border region labels to zero for random walker
        ## segmentation in next step
        np.place(self.labeled, np.logical_not(self.skeletonized), 0) ## updates in place

        ## Segmentation of border pixels
        self.final = segmentation.random_walker(self.skeletonized, self.labeled, beta=1,  mode='cg_mg')

def load_image(buf):
    """Load image from buffer into numpy array"""
    img_bytes = buf.read()
    pil_image = PIL.Image.open(io.BytesIO(img_bytes))
    return(np.asarray(pil_image))

def save(self, im, cmap):
    """Save image to a buffer"""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(im, cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, dpi=300, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return(buf)

def plt_image(im, cmap='gray'):
    """plot image using matplotlib"""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(im, cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
