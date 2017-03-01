import tensorflow as tf
import os

class Network():
    
    def __init__(self, nrOfGPU = 1):
        config = tf.ConfigProto(device_count = {"GPU" : nrOfGPU});
        config.gpu_options.allow_growth = True;
        self.sess = tf.InteractiveSession(config = config);
        
        self.inputHeight = 52;
        self.inputWidth = 52;
        self.channels = 3;
        
        self.nrOfClasses = 2;
        
        self.lr = tf.placeholder(tf.float32);
        self.input = tf.placeholder(tf.float32, shape = [None, self.inputHeight, self.inputWidth, self.channels]);
        self.target = tf.placeholder(tf.float32, shape = [None, self.nrOfClasses]);
        self.keep_prop = tf.placeholder(tf.float32);
    
    def WeightVariable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.1);
        return tf.Variable(initial);
    
    def BiasVariable(self, shape):
        initial = tf.constant(0.1, shape = shape);
        return tf.Variable(initial);
    
    def Conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME");
    
    def MaxPool2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1,2,2,1], padding = "SAME");
    
    def CreateNetwork(self):

        convolutions = [[9,9,self.channels, 16], "pool", [3,3,16,16], "pool"];
        
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

                
        wi = 0;
        for i in range(len(convolutions)):
            
            if convolutions[i] == "pool":
                self.convOutput.append(self.MaxPool2x2(self.convOutput[-1]));
                        
            elif i == 0:
                outShape = convolutions[i][-1];
                beta = tf.Variable(tf.constant(0.0, shape=[outShape]), name='beta', trainable=True)
                gamma = tf.Variable(tf.constant(1.0, shape=[outShape]), name='gamma', trainable=True)
                convOutput = self.Conv2d(self.input, self.weightsConv[wi]) + self.biasConv[wi];
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
        nrOfConvFeatures = finalNrOfKernels * 13 * 13 * 1
        print "Number of CNN Features: ", nrOfConvFeatures;
        self.flatShape = tf.reshape(self.convOutput[-1], [-1, nrOfConvFeatures]);
        
        self.topLayers = [200];
        
        self.weight = [];
        self.bias = [];

        
        for i in range(0, len(self.topLayers)):
            if (i == len(self.topLayers) - 1): # Last layer so output size
                if (len(self.topLayers) > 1):
                    self.weight.append(self.WeightVariable([self.topLayers[i-1], self.nrOfClasses]));
                else:
                    self.weight.append(self.WeightVariable([nrOfConvFeatures, self.nrOfClasses]));
                self.bias.append(self.BiasVariable([self.nrOfClasses]));
                
            elif (i == 0):
                self.weight.append(self.WeightVariable([nrOfConvFeatures, self.topLayers[i]]));
                self.bias.append(self.BiasVariable([self.topLayers[i]]));
                
            else:
                self.weight.append(self.WeightVariable([self.topLayers[i-1], self.topLayers[i]]));
                self.bias.append(self.BiasVariable([self.topLayers[i]]));
                
                
        self.topLayers = [];
   
        
        for i in range(0, len(self.weight)):            
            if (i == 0 and i == len(self.weight) - 1):
                self.topLayers.append(tf.add(tf.matmul(self.flatShape, self.weight[i]), self.bias[i]));
            
            elif (i == 0): # First layer, takes flatShape as input (which is the output of Conv layers)
                self.topLayers.append(tf.nn.relu6(tf.add(tf.matmul(self.flatShape, self.weight[i]), self.bias[i])));
            
            elif (i == len(self.weight) - 1): # Final layer
                self.topLayers.append(tf.add(tf.matmul(self.topLayers[-1], self.weight[i]), self.bias[i]));
            else: # Connect the rest of the top layers
                self.topLayers.append(tf.nn.relu6(tf.add(tf.matmul(self.topLayers[-1], self.weight[i]), self.bias[i])));       
                
        
        self.topOutput = self.topLayers[-1];
        self.dropoutLayer = tf.nn.dropout(self.topOutput, keep_prob = self.keep_prop);
        self.cross_entropy_run = tf.nn.softmax(self.dropoutLayer);
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.topOutput, labels = self.target));
        
        self.trainX = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy);
        
        self.correctPrediction = tf.equal(tf.argmax(self.target, 1), tf.argmax(self.topOutput, 1));
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPrediction, tf.float32))

        
        self.saver = tf.train.Saver();
        self.saverConv = tf.train.Saver(weightDict);
        self.sess.run(tf.global_variables_initializer());  
        
    def Train(self, input, target, lr):        
        self.trainX.run(session = self.sess, feed_dict = {self.input : input, self.target: target, self.lr : lr, self.keep_prop : 0.5});     
        
    def Save(self, path, param = None):
        if param == None:
            model = "cnn_model.ckpt";
            modelConv = "cnn_conv_model.ckpt";
        else:
            model = "cnn_model" + param + ".ckpt";
            modelConv = "cnn_conv_model" + param + ".ckpt";
        self.saver.save(self.sess, os.path.join(path, model));
        self.saverConv.save(self.sess, os.path.join(path, modelConv)); 
        
    def Load(self, path, model = "cnn_model.ckpt"):
        self.saver.restore(self.sess, os.path.join(path, model));
        
    def LoadConv(self, path, model = "cnn_model_conv.ckpt"):
        self.saverConv.restore(self.sess, os.path.join(path, model));
        
    def Run(self, input):
        return self.cross_entropy_run.eval(session = self.sess, feed_dict = {self.input : input, self.keep_prop : 1.0});
    
    def Eval(self, input, target):
        return self.accuracy.eval(session = self.sess, feed_dict = {self.input : input, self.target : target, self.keep_prop : 1.0});
    
            
                            
            
                
                
