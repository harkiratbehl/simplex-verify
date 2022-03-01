###################################################################
####### ORGINAL FILE FROM https://github.com/eth-sri/eran# ########
############# MODIFIED by: ########################################

# import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import re


def product(it):
    product = 1
    for x in it:
        product *= x
    return product

def runRepl(arg, repl):
    for a in repl:
        arg = arg.replace(a+"=", "'"+a+"':")
    return eval("{"+arg+"}")

def extract_mean(text):
    mean = ''
    m = re.search('mean=\[(.+?)\]', text)
    
    if m:
        means = m.group(1)
    mean_str = means.split(',')
    num_means = len(mean_str)
    mean_array = np.zeros(num_means)
    for i in range(num_means):
         mean_array[i] = np.float64(mean_str[i])
    return mean_array

def extract_std(text):
    std = ''
    m = re.search('std=\[(.+?)\]', text)
    if m:
        stds = m.group(1)
    std_str =stds.split(',')
    num_std = len(std_str)
    std_array = np.zeros(num_std)
    for i in range(num_std):
        std_array[i] = np.float64(std_str[i])
    return std_array

def numel(x):
    return product([int(i) for i in x.shape])

def parseVec(net):
    return np.array(eval(net.readline()[:-1]))

def myConst(vec):
    return tf.constant(vec.tolist(), dtype = tf.float64)

def myConstp(vec):
    return vec

def permutation(W, h, w, c):
    m = np.zeros((h*w*c, h*w*c))
    
    column = 0
    for i in range(h*w):
        for j in range(c):
            m[i+j*h*w, column] = 1
            column += 1
    
    return np.matmul(W, m) 

def read_eran_tf_net(net_file, in_len):
    mean = 0.0
    std = 0.0
    net = open(net_file,'r')
    x = tf.placeholder(tf.float64, [in_len], name = "x")
    y = None
    z1 = None
    z2 = None
    last_layer = None
    h,w,c = None, None, None
    is_conv = False
    w_b_list=[]
    while True:
        curr_line = net.readline()[:-1]
        if 'Normalize' in curr_line:
            mean = extract_mean(curr_line)
            std = extract_std(curr_line)
        elif curr_line in ["ReLU", "Affine"]:
            print(curr_line)
            W = None
            if (last_layer in ["Conv2D"]):
                wi=permutation(parseVec(net), h, w, c).transpose()
                W = myConst(wi)
                Wp = myConstp(wi)
            else:
                wi=parseVec(net).transpose()
                W = myConst(wi)
                Wp = myConstp(wi)
            bi = parseVec(net)
            #b = myConst(b.reshape([1, numel(b)]))
            b = myConst(bi)
            bp = myConstp(bi)
            if(curr_line=="Affine"):
                x = tf.nn.bias_add(tf.matmul(tf.reshape(x, [1, numel(x)]),W), b)
            elif(curr_line=="ReLU"):
                x = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.reshape(x, [1, numel(x)]),W), b))
            print("\tOutShape: ", x.shape)
            print("\tWShape: ", W.shape)
            print("\tBShape: ", b.shape)
            w_b_list.append(["Linear", Wp, bp])
        elif curr_line == "Conv2D":
            is_conv = True
            line = net.readline()
            args = None
            #print(line[-10:-3])
            start = 0
            if("ReLU" in line):
                start = 5
            elif("Affine" in line):
                start = 7
            if 'padding' in line:
                args =  runRepl(line[start:-1], ["filters", "input_shape", "kernel_size", "stride", "padding"])
            else:
                args = runRepl(line[start:-1], ["filters", "input_shape", "kernel_size"])

            wi = parseVec(net)
            W = myConst(wi)
            Wp = myConstp(wi)
            print("W shape", W.shape)
            #W = myConst(permutation(parseVec(net), h, w, c).transpose())
            b = None
            if("padding" in line):
                if(args["padding"]>=1):
                    padding_arg = "SAME"
                else:
                    padding_arg = "VALID"
            else:
                padding_arg = "VALID"

            if("stride" in line):
                stride_arg = [1] + args["stride"] + [1]
            else:
                stride_arg = [1,1,1,1]
            # input('')
            x = tf.nn.conv2d(tf.reshape(x, [1] + args["input_shape"]), filter=W, strides=stride_arg, padding=padding_arg)

            bi = parseVec(net)
            b = myConst(bi)
            bp = myConstp(bi)
            h, w, c = [int(i) for i in x.shape ][1:]
            print("Conv2D", args, "W.shape:",W.shape, "b.shape:", b.shape)
            print("\tOutShape: ", x.shape)
            w_b_list.append([curr_line, Wp, bp])
            # if("ReLU" in line):
            #     x = tf.nn.relu(tf.nn.bias_add(x, b))
            # elif("Sigmoid" in line):
            #     x = tf.nn.sigmoid(tf.nn.bias_add(x, b))
            # elif("Tanh" in line):
            #     x = tf.nn.tanh(tf.nn.bias_add(x, b))
            # elif("Affine" in line):
            #     x = tf.nn.bias_add(x, b)
            # else:
            #     raise Exception("Unsupported activation: ", curr_line)
        elif curr_line == "":
            break
        else:
            raise Exception("Unsupported Operation: ", curr_line)
        last_layer = curr_line

    model = x
    return model, is_conv, mean, std, w_b_list
