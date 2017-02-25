import tensorflow as tf
import os

class Network():
    
    def __init__(self, nrOfGPU = 1):
        config = tf.ConfigProto(device_count = {"GPU" : nrOfGPU});
        config.gpu_options.allow_growth = True;
        self.sess = tf.InteractiveSession(config = config);
        
        self.inputHeight = 28;
        self.inputWidth = 28;
        self.channels = 1;
        
        self.nrOfClasses = 10;
        
        self.lr = tf.placeholder(tf.float32);
        self.input = tf.placeholder(tf.float32, shape = [None, self.inputHeight * self.inputWidth * self.channels]);
        self.target = tf.placeholder(tf.float32, shape = [None, self.nrOfClasses]);
        self.keep_prop = tf.placeholder(tf.float32);
    
    def WeightVariable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01);
        return tf.Variable(initial);
    
    def BiasVariable(self, shape):
        initial = tf.constant(0.01, shape = shape);
        return tf.Variable(initial);
    
    def Conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME");
    
    def MaxPool2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = "SAME");
    
    def CreateNetwork(self):

        convolutions = [[9,9,self.channels, 32],[7,7,32, 64], [5, 5, 64, 128], "pool", [3,3,128,64],[3,3,64,32], [3,3,32,32], [3,3,32,16], "pool"];

        
        self.weightsConv = [];
        self.biasConv = [];
        
        
        for i in range(len(convolutions)):
            shape = convolutions[i];
            if (shape == "pool"):
                pass
                #self.weightsConv.append("pool");
                #self.biasConv.append("pool");
            else:
                self.weightsConv.append(self.WeightVariable(shape));
                self.biasConv.append(self.BiasVariable([shape[-1]]));
                
        weightDict = {};    
        for i in range(len(self.weightsConv)):
            weightDict['w' + str(i)] = self.weightsConv[i];
            weightDict['b' + str(i)] = self.biasConv[i];
         
        self.convOutput = [];   

        self.flatInput = tf.reshape(self.input, [-1, self.inputHeight, self.inputWidth, self.channels])
        
        wi = 0;
        for i in range(len(convolutions)):
            
            if convolutions[i] == "pool":
                self.convOutput.append(self.MaxPool2x2(self.convOutput[-1]));
                        
            elif i == 0:
                outShape = convolutions[i][-1];
                beta = tf.Variable(tf.constant(0.0, shape=[outShape]), name='beta', trainable=True)
                gamma = tf.Variable(tf.constant(1.0, shape=[outShape]), name='gamma', trainable=True)
                convOutput = self.Conv2d(self.flatInput, self.weightsConv[wi]) + self.biasConv[wi];
                batch_mean, batch_var = tf.nn.moments(convOutput, [0,1,2], name='moments')
                batchNormal = tf.nn.batch_normalization(convOutput, batch_mean, batch_var, beta, gamma, 1e-3);
                self.convOutput.append(tf.nn.relu6(batchNormal));    
                wi += 1;                        
            else:
                outShape = convolutions[i][-1];
                beta = tf.Variable(tf.constant(0.0, shape=[outShape]), name='beta', trainable=True)
                gamma = tf.Variable(tf.constant(1.0, shape=[outShape]), name='gamma', trainable=True)
                convOutput = self.Conv2d(self.convOutput[-1], self.weightsConv[wi]) + self.biasConv[wi];
                batch_mean, batch_var = tf.nn.moments(convOutput, [0,1,2], name='moments')
                batchNormal = tf.nn.batch_normalization(convOutput, batch_mean, batch_var, beta, gamma, 1e-3);
                self.convOutput.append(tf.nn.relu6(batchNormal));
                wi += 1;
                
        shape = convolutions[-2];
        finalNrOfKernels = shape[-1];
        
        # Output of the Convolution part, flatted
        nrOfConvFeatures = finalNrOfKernels * 7 * 7 * 1
        print "Number of CNN Features: ", nrOfConvFeatures;
        self.flatShape = tf.reshape(self.convOutput[-1], [-1, nrOfConvFeatures]);
                        

        self.saverConv = tf.train.Saver(weightDict);
        self.sess.run(tf.global_variables_initializer());  
          
    
        
    def LoadConv(self, path, model = "cnn_model_conv.ckpt"):
        self.saverConv.restore(self.sess, os.path.join(path, model));
        
    def Run(self, input):
        return self.flatShape.eval(session = self.sess, feed_dict = {self.input : input, self.keep_prop : 1.0});
    
                            
            
                
                
