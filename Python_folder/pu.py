import tensorflow as tf
import find_countours as F_C
#from PIL import Image
height  = 128
weight = 128

filename1 = '/home/artem/Documents/Projects/4.jpg' # name of image to export
filename2 = '/home/artem/Documents/Projects/4_2.jpeg'
filename3 = '/home/artem/Documents/Projects/4_1.jpg'
#img = Image.open(filename1)
#img.show()
#F_C.Show_image(filename1)

#F_C.Countours_of_image(filename3)

def Resizing(tensor_name, size1, size2):
  return tf.image.resize_images(tensor_name,[size1, size2])

def Take_tensors(filename):# take tensors from image
  image_tf_string = tf.read_file(filename) # need to go name of image to tf format of name
  image_tens=tf.image.decode_jpeg(contents=image_tf_string , channels=1) #rgb decoding 
  image_resized = Resizing(image_tens,height,weight)
  print (sess.run(image_resized))
  return image_resized  #return existed tensor

def Loses_print(file1, file2): #difference of images
  print ('Next')
  losses =  tf.losses.mean_squared_error(labels=Take_tensors(file1), predictions=Take_tensors(file2))
  print(sess.run(losses))
  return losses


def Write_graphs(): #add graphs
  writer = tf.summary.FileWriter('event_image/'+'.')
  writer.add_graph(tf.get_default_graph())



def Go_optim():
  loss = tf.losses.mean_squared_error(labels=Take_tensors(filename1), predictions=Take_tensors(filename2))
  optimizer= tf.train.GradientDescentOptimizer(0.01)
  train=optimizer.minimize(loss)
  for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(loss_value)

if __name__ == "__main__":
  sess =tf.Session() # run tf.Session
 # print (sess.run(Take_tensors(filename1))) #print output Tensor
  print('LOSES')
  #print (sess.run(Take_tensors(filename2))) #print output Tensor
  Loses_print(filename1, filename2)
  Loses_print(filename1,filename3)
  Loses_print(filename2, filename3)
  Write_graphs()
  
"""
def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized

sess = tf.Session()

# A vector of filenames.
filenames = tf.constant([filename1])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([0])

dataset = tf.data.Dataset.from_tensor_slices((filename1))
dataset = dataset.map(_parse_function)

dataset.cache()
"""