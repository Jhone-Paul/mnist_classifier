import numpy as np

class Conv3x3:
    def __init__ (self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3)/9

    def iterate_regions(self, image):
        h, w = image.shape

        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region,i,j
    def forward(self, input):

        self.last_input = input

        h, w = input.shape
        output = np.zeros((h-2,w-2,self.num_filters))

        # iterate_regions() is a helper generator method that yields all valid 3x3 image regions
        for im_region, i, j in self.iterate_regions(input):
            output[i,j] =np.sum(im_region *self.filters, axis =(1,2))

        return output

    def backprop(self, d_L_d_out, learn_rate):
        
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        # We aren't returning anything here since we use Conv3x3 as
        # the first layer in our CNN. Otherwise, we'd need to return
        # the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
        return None
