#prject DEEP LEARNING#
library(reticulate)
library(keras)
np = import("numpy")

names_control_list = readLines("../data/control_list.txt")
names_list = readLines("../data/list.txt")
names_nucleotide_list = readLines("../data/nucleotide_list.txt")
names_steroid_list = readLines("../data/steroid_list.txt")

n_struct_try = 100

#x_test
x_train = array(data = NA, dim = c(n_struct_try*2,14,32,32,32))
y_train = array(data = 0, dim = c(n_struct_try*2,2))
for (i in seq(1,n_struct_try*2,2)) {
  structure_nucleotide = np$load(paste0(paste('../data/deepdrug3d_voxel_data/', names_nucleotide_list[i],sep = ""),".npy"))
  structure_control = np$load(paste0(paste('../data/deepdrug3d_voxel_data/', names_list[i],sep = ""),".npy"))
  x_train[i,,,,] = structure_nucleotide
  y_train[i,1] = 1
  x_train[i+1,,,,] = structure_control
  y_train[i+1,2] = 1
}

#### MODEL ####
model = keras_model_sequential()

layer_conv_3d(model, filters = 32 , kernel_size = c(3,3,3), activation = "relu", input_shape = c(14,32,32,32), data_format='channels_first')

layer_dropout(model, 0.3)
layer_flatten(model)

layer_dense(model, units = 2, activation = "softmax")

compile(model,loss = 'categorical_crossentropy',optimizer = optimizer_rmsprop(), metrics = "accuracy")
history = fit(model, x_train, y_train, epochs = 20, batch_size = 32, validation_split = 0.8)
evaluate(model, x_test, y_test)
plot(history)
####

