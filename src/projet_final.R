#DEEP LEARNING project#
library(reticulate)
library(keras)
np = import("numpy")
library(gplots)
library(ROCR)

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
prct_struct_heme = 0.2 #597 structures
prct_struct_nucleotide = 0.2 #1554 structures
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
y_train = array(data = 0, dim = c(n_struct_train,3))
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
y_test = array(data = 0, dim = c(n_struct_test,3))
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
#model 1 - model article
#model = keras_model_sequential()
#layer_conv_3d(model, filters = 64 , kernel_size = 5, activation = "relu", input_shape = c(14,32,32,32), data_format='channels_first')
#layer_dropout(model, 0.2)
#layer_conv_3d(model, filters = 64 , kernel_size = 3, activation = "relu", data_format='channels_first')
#layer_max_pooling_3d(model, pool_size = c(2,2,2), data_format='channels_first')
#layer_dropout(model, 0.4)
#layer_flatten(model)
#layer_dense(model, units = 128, activation = "relu")
#layer_dropout(model, 0.4)
#layer_dense(model, units = 3, activation = "softmax")
#
#modele 2 perso :
model = keras_model_sequential()
layer_conv_3d(model, filters = 32 , kernel_size = c(3,3,3), activation = "relu", input_shape = c(14,32,32,32), data_format='channels_first')
layer_conv_3d(model, filters = 64 , kernel_size = c(3,3,3), activation = "relu", data_format='channels_first')
layer_conv_3d(model, filters = 128 , kernel_size = c(3,3,3), activation = "relu", data_format='channels_first')
layer_max_pooling_3d(model, pool_size = c(3,3,3), data_format='channels_first')
layer_dropout(model, 0.2)
layer_max_pooling_3d(model, pool_size = c(3,3,3), data_format='channels_first') 
layer_dropout(model, 0.4)
layer_flatten(model)

layer_dense(model, units = 3, activation = "softmax")
#
#modele 3 
#input = layer_input(shape = c(14,32,32,32))
#output = layer_conv_3d(input, filters = 8, kernel_size = c(3,3,3), padding = "same", 
#                           activation = "relu", kernel_initializer = "he_normal", data_format='channels_first')
#for(i in 1:2){
#  save = output
#  if(save$shape[-1] != 8){
#    output = layer_conv_3d(output, filters = 8, kernel_size = c(3,3,3), padding = "same", 
#                           activation = "relu", kernel_initializer = "he_normal", data_format='channels_first')
#  }
#  output1 = layer_conv_3d(output, filters = 4, kernel_size = c(3,3,3), activation = "relu",
#                          padding = "same", kernel_initializer = "he_normal", data_format='channels_first')
#  output2 = layer_conv_3d(output1, filters = 8, kernel_size = c(3,3,3), activation = "linear",
#                          padding = "same", kernel_initializer = "he_normal", data_format='channels_first')
#  conc = layer_add(inputs = list(output2, save))
#  activ = layer_activation(conc, activation = "relu")
#  output = activ
#}
#mod = output
#model_max1 = layer_max_pooling_3d(mod, pool_size = c(3,3,3), data_format='channels_first') 
#model_drop1 = layer_dropout(model_max1, 0.4)
#model_flatten1 = layer_flatten(model_drop1)
#model_dense1 = layer_dense(model_flatten1, units = 3, activation = "softmax")
#model = keras_model(inputs = input, outputs = model_dense1)
#

compile(model,loss = 'categorical_crossentropy',optimizer = optimizer_adam(), metrics = "accuracy")
history = fit(model, x_train, y_train, epochs = 20, batch_size = 32, validation_split = 0.3)
evaluate(model, x_test, y_test)
plot(history)
####
#### Performances on test####
#
model_accuracy = function(model, x_test, y_test, names_test){
  cat("Accuracy :","\n")
  print(evaluate(model, x_test, y_test))
  
  model_predict_test  = predict(model, x_test)
  model_predict_test_val = NULL
  model_predict_test_val = max.col(model_predict_test)
  pred = factor(model_predict_test_val, 1:3)
  test = factor(max.col(y_test), 1:3)
  TableTot = table(pred, test)
  
  cat("Table :","\n")
  cat(TableTot,"\n")
  cat("List of pockets badly predicted :","\n")
  cat("Classes :","\n")
  cat("1 : control pockets","\n")
  cat("2 : nucleotide-binding pockets","\n")
  cat("3 : heme-binding pockets","\n")
  cat("Pockets Predicted Reality","\n")
  for (i in 1:length(pred)) {
    if(pred[i] != test[i]){
      cat(">", names_test[i], pred[i], test[i],"\n")
    }
  }
}
model_negativ_test = function(model, names_steroid_list){
  x_test = array(data = NA, dim = c(length(names_steroid_list), 14, 32, 32, 32))
  names_test = c(names_steroid_list)
  for (i in 1:length(names_test)) {
    x_test[i, , , , ] = np$load(paste0(paste('../data/deepdrug3d_voxel_data/', names_test[i],sep = ""), ".npy"))
  }
  return(x_test)
}

## ROC CURVE ##
model_ROC_curve = function(model, x_test, y_test){
  model_predict_test  = predict(model, x_test)
  
  n <- 3 # you have n models
  color_style <- c('red', 'blue',"green") # 2 colors
  for (i in 1:n) {
    plot(performance(prediction(model_predict_test[,i],y_test[,i]),"tpr","fpr"), 
         add=(i!=1),col=color_style[i],lwd=2, cex.lab=1.5)
  }
  title("ROC curve on validation data set")
  dt.auc = performance(prediction(model_predict_test,y_test), "auc")
  abline(0,1)
  legend("bottomright", c("control pockets", "nucleotide-binding pockets", "heme-binding pockets"), 
          col=color_style, lty=c(1,1,1), cex=0.8)
  cat("----AUC VALUES OF THE MODEL----")
  print(attr(dt.auc, "y.values"))
}
#### Tests ####

#
x_test_steroid  = model_negativ_test(model, names_steroid_list)
categories = c(rep(0,length(names_steroid_list)))
y_test_steroid = to_categorical(categories,3)

#
#model = load_model_hdf5("../results/final_2.h5")
sink("../results/final_2.txt")
model_accuracy(model, x_test, y_test, names_test)
print("---NEGATIVE CONTROL FOR THE STEROIDS---")
model_accuracy(model, x_test_steroid, y_test_steroid)
jpeg("../results/final_2_roc.jpeg") 
model_ROC_curve(model, x_test, y_test)
dev.off()
sink()
save_model_hdf5(model, "../results/model.h5")







