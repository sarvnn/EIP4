# EIP4
Learning DNN

# Assignment:
Change the model in such a way that after executing the code below, your accuracy print out is more than 99.0
        score = model.evaluate(X_text, Y_text, verbose=0)
        print(score)
        
        [0.023546614046123023, 0.995]
        
# Write your own definitions for the following:
        
        //Please disregard the commented sections marked by double slashes. they are for my understanding //
        
        Convolution:
        
        Convolution is a set of matrix transformations performed piecewise(usually a smaller square matrix) on a multidimensional input data by overlapping the pieces with a predefined matrix of arbitrary size called as the kernel. 
        
        ///The output obtained is specifically defined by the kernel used. Convolution is useful in bringing out spatial features from multidimensional data such as images.///
        
        Filters/Kernels:
        
        Kernels/ Filters are the predefined matrix of arbitrary size, which can filter specific patterns/details present in the input data (such as a horizontal edge, or a gradient) when used to perform the piecewise matrix transformation on the input.
            
        Epochs:
                
        When a neural network is allowed to learn through the entirety of the training dataset once, it is termed as one epoch. It includes a foward run and a reverse run.
        
        //As opposed to a batch which refers to the number of datasets taken per forward and backward pass//
        
        1x1 Convolution:
        
        Here the kernel size is limited to 1 along two dimensions. This is useful in obtaining single pixel information from multiple channels (or layers).
        
        //1x1 convolution is typically used to extract patterns from between different channels like colors//
        
        3x3 Convolution:
        
        Here the kernel size is 3 along two dimensions. This is useful in collating patterns from a local spatial piece of the input data. Multiple 3x3 convolutions are necessary to obtain patterns present in larger spaces of the input.
        
        Feature Maps:
        
        Feature maps are individual channels obtained after convolution using a kernel. After convolution, based on the kernel used, distinct patterns/details obtained from the original input will be present in the feature map. 
        
        
        //we will get as many feature maps as the number of filters/kernels we will use on the input.// 
        
        
        Activation Function:
        
        Different neurons of the neural network carry different information upon learning from the input dataset. The activation function selects which of the neurons will operate to make a decision on the input signal.
        
        //
        
        
        Receptive Field:
        
       The receptive field is the piece of the input data that is available for a kernel to do the matrix operation towards convolution.
       
       //It is the area of the image that a kernel can see and thus perceive patterns. //
       
