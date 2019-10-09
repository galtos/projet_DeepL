#prject DEEP LEARNING#
library(reticulate)
library(keras)
np = import("numpy")

names_control_list = readLines("../data/control_list.txt")
names_heme_list = readLines("../data/list.txt")
names_nucleotide_list = readLines("../data/nucleotide_list.txt")
names_steroid_list = readLines("../data/steroid_list.txt")

names_control_list = names_control_list[-which(names_control_list == "")]
names_heme_list = names_heme_list[-which(names_heme_list == "")]
names_nucleotide_list = names_nucleotide_list[-which(names_nucleotide_list == "")]
names_steroid_list = names_steroid_list[-which(names_steroid_list == "")]
#
prct_struct_control = 0.05 #1947 structures
prct_struct_heme = 0.1 #597 structures
prct_struct_nucleotide = 0.1 #1554 structures
prct_struct_steroid = 0 #70 structures
#
prct_train = 2/3
prct_test = 1/3
#
Index_control_train = sample(1:length(names_control_list),size = length(names_control_list)*prct_train*prct_struct_control )
Index_heme_train = sample(1:length(names_heme_list),size = length(names_heme_list)*prct_train*prct_struct_heme )
Index_nucleotide_train = sample(1:length(names_nucleotide_list),size = length(names_nucleotide_list)*prct_train*prct_struct_nucleotide )
Index_steroid_train = sample(1:length(names_steroid_list),size = length(names_steroid_list)*prct_train*prct_struct_steroid )
#
Index_control_test = sample(1:length(names_control_list[-Index_control_train]),size = length(names_control_list[-Index_control_train])*prct_test*prct_struct_control )
Index_heme_test = sample(1:length(names_heme_list[-Index_heme_train]),size = length(names_heme_list[-Index_heme_train])*prct_test*prct_struct_heme )
Index_nucleotide_test = sample(1:length(names_nucleotide_list[-Index_nucleotide_train]),size = length(names_nucleotide_list[-Index_nucleotide_train])*prct_test*prct_struct_nucleotide )
Index_steroid_test = sample(1:length(names_steroid_list[-Index_steroid_train]),size = length(names_steroid_list[-Index_steroid_train])*prct_test*prct_struct_steroid )
#
names_control_train = names_control_list[Index_control_train]
names_heme_train = names_heme_list[Index_heme_train]
names_nucleotide_train = names_nucleotide_list[Index_nucleotide_train]
names_steroid_train = names_steroid_list[Index_steroid_train]
#
names_control_test = names_control_list[Index_control_test]
names_heme_test = names_heme_list[Index_heme_test]
names_nucleotide_test = names_nucleotide_list[Index_nucleotide_test]
names_steroid_test = names_steroid_list[Index_steroid_test]
#
n_struct_train = length(names_control_train)+length(names_heme_train)+length(names_nucleotide_train)+length(names_steroid_train)
n_struct_test = length(names_control_test)+length(names_heme_test)+length(names_nucleotide_test)+length(names_steroid_test)
####x_train et y_train####
x_train = array(data = NA, dim = c(n_struct_train,14,32,32,32))
y_train = array(data = 0, dim = c(n_struct_train,4))
j = 1
names = c(names_control_train,names_heme_train,names_nucleotide_train,names_steroid_train)
for (i in 1:n_struct_train) {
  x_train[i,,,,] = np$load(paste0(paste('../data/deepdrug3d_voxel_data/', names[i],sep = ""),".npy"))
  y_train[i,j] = 1
  if(i == length(names_control_train)) {
    j = j+1
  }
  if(i == length(names_control_train)+length(names_heme_train)) {
    j = j+1
  }
  if(i == length(names_control_train)+length(names_heme_train)+length(names_nucleotide_train)) {
    j = j+1
  }
  if(i == length(names_control_train)+length(names_heme_train)+length(names_nucleotide_train)+length(names_steroid_train)) {
    j = j+1
  }
}
####x_test et y_test####
x_test = array(data = NA, dim = c(n_struct_test,14,32,32,32))
y_test = array(data = 0, dim = c(n_struct_test,4))
j = 1
names_test = c(names_control_test,names_heme_test,names_nucleotide_test,names_steroid_test)
for (i in 1:n_struct_test) {
  x_test[i,,,,] = np$load(paste0(paste('../data/deepdrug3d_voxel_data/', names_test[i],sep = ""),".npy"))
  y_test[i,j] = 1
  if(i == length(names_control_test)) {
    j = j+1
  }
  if(i == length(names_control_test)+length(names_heme_test)) {
    j = j+1
  }
  if(i == length(names_control_test)+length(names_heme_test)+length(names_nucleotide_test)) {
    j = j+1
  }
  if(i == length(names_control_test)+length(names_heme_test)+length(names_nucleotide_test)+length(names_steroid_test)) {
    j = j+1
  }
}
#### MODEL ####
#attention à mettre - data_format='channels_first' - partout
model = keras_model_sequential()

layer_conv_3d(model, filters = 32 , kernel_size = c(3,3,3), activation = "relu", input_shape = c(14,32,32,32), data_format='channels_first')
layer_conv_3d(model, filters = 64 , kernel_size = c(3,3,3), activation = "relu", data_format='channels_first')

layer_conv_3d(model, filters = 128 , kernel_size = c(3,3,3), activation = "relu", data_format='channels_first')
layer_max_pooling_3d(model, pool_size = c(3,3,3), data_format='channels_first')

layer_dropout(model, 0.2)
layer_max_pooling_3d(model, pool_size = c(3,3,3), data_format='channels_first') 
layer_dropout(model, 0.4)
layer_flatten(model)

layer_dense(model, units = 4, activation = "softmax")
#
compile(model,loss = 'categorical_crossentropy',optimizer = optimizer_adam(), metrics = "accuracy")
history = fit(model, x_train, y_train, epochs = 20, batch_size = 32, validation_split = 0.3)
evaluate(model, x_test, y_test)
plot(history)
####
model_predict_test  = predict(model, x_test)
model_predict_test_val = NULL


model_predict_test_val = max.col(model_predict_test)


##accuracy test

pred = factor(model_predict_test_val, 1:4)
test = factor(max.col(y_test), 1:4)
TableTot = table(pred, test)

accuracy = sum(diag(TableTot))/sum(TableTot)
accuracy

## ROC CURVE ##


####save model####
save_model_hdf5(model, "model_test_1.h5")
####test vrai model####
model = load_model_hdf5("deepdrug3d.h5")







