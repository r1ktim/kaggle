import tensorflow as tf
import os

class Network():
    
    def __init__(self, nrOfGPU = 1):
        config = tf.ConfigProto(device_count = {"GPU" : nrOfGPU});
        config.gpu_options.allow_growth = True;
        self.sess = tf.InteractiveSession(config = config);
        
        self.nrOfInputs = 784;        
        self.nrOfClasses = 10;
        
        self.lr = tf.placeholder(tf.float32);
        self.input = tf.placeholder(tf.float32, shape = [None, self.nrOfInputs]);
        self.target = tf.placeholder(tf.float32, shape = [None, self.nrOfClasses]);
        self.keep_prop = tf.placeholder(tf.float32);
    
    def WeightVariable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01);
        return tf.Variable(initial);
    
    def BiasVariable(self, shape):
        initial = tf.constant(0.01, shape = shape);
        return tf.Variable(initial);
    
    def CreateNetwork(self):
                
        self.topLayers = [50, 500, 50];
        
        self.weight = [];
        self.bias = [];
        
        for i in range(0, len(self.topLayers)):
            if (i == len(self.topLayers) - 1): # Last layer so output size
                if (len(self.topLayers) > 1):
                    self.weight.append(self.WeightVariable([self.topLayers[i-1], self.nrOfClasses]));
                else:
                    self.weight.append(self.WeightVariable([self.nrOfInputs, self.nrOfClasses]));
                self.bias.append(self.BiasVariable([self.nrOfClasses]));
                
            elif (i == 0):
                self.weight.append(self.WeightVariable([self.nrOfInputs, self.topLayers[i]]));
                self.bias.append(self.BiasVariable([self.topLayers[i]]));
                
            else:
                self.weight.append(self.WeightVariable([self.topLayers[i-1], self.topLayers[i]]));
                self.bias.append(self.BiasVariable([self.topLayers[i]]));
                
                
        self.topLayers = [];
   
        
        for i in range(0, len(self.weight)):            
            if (i == 0 and i == len(self.weight) - 1):
                self.topLayers.append(tf.add(tf.matmul(self.input, self.weight[i]), self.bias[i]));
            
            elif (i == 0): # First layer, takes flatShape as input (which is the output of Conv layers)
                self.topLayers.append(tf.nn.relu6(tf.add(tf.matmul(self.input, self.weight[i]), self.bias[i])));
            
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
        self.sess.run(tf.global_variables_initializer());  
        
    def Train(self, input, target, lr):        
        self.trainX.run(session = self.sess, feed_dict = {self.input : input, self.target: target, self.lr : lr, self.keep_prop : 0.5});     
        
    def Save(self, path, param = None):
        if param == None:
            model = "top_model.ckpt";
        else:
            model = "top_model" + param + ".ckpt";

        self.saver.save(self.sess, os.path.join(path, model));

        
    def Load(self, path, model = "top_model.ckpt"):
        self.saver.restore(self.sess, os.path.join(path, model));
        
    def LoadConv(self, path, model = "cnn_model_conv.ckpt"):
        self.saverConv.restore(self.sess, os.path.join(path, model));
        
    def Run(self, input):
        return self.cross_entropy_run.eval(session = self.sess, feed_dict = {self.input : input, self.keep_prop : 1.0});
    
    def Eval(self, input, target):
        return self.accuracy.eval(session = self.sess, feed_dict = {self.input : input, self.target : target, self.keep_prop : 1.0});
    
            
                            
            
                
                
