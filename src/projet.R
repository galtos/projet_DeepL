#prject DEEP LEARNING#
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
prct_struct_control = 0.16#0.05 #1947 structures
prct_struct_heme = 0.5#0.1 #597 structures
prct_struct_nucleotide = 0.2#0.1 #1554 structures
#steroid dans le test seulement
prct_struct_steroid = 0 #70 structures
#
prct_train = 2/3
prct_test = 1/3
#
set.seed(61565)
#
Index_control_train = sample(1:length(names_control_list),size = length(names_control_list)*prct_train*prct_struct_control )
Index_heme_train = sample(1:length(names_heme_list),size = length(names_heme_list)*prct_train*prct_struct_heme )
Index_nucleotide_train = sample(1:length(names_nucleotide_list),size = length(names_nucleotide_list)*prct_train*prct_struct_nucleotide )
Index_steroid_train = sample(1:length(names_steroid_list),size = length(names_steroid_list)*0*prct_struct_steroid ) #set to 0 for all steroid data in tests
#
size_train = length(Index_control_train) + length(Index_heme_train) + length(Index_nucleotide_train) + length(Index_steroid_train)
length(Index_control_train)
length(Index_heme_train)
length(Index_nucleotide_train)
length(Index_steroid_train)
#
Index_control_test = sample(1:length(names_control_list[-Index_control_train]),size = length(names_control_list[-Index_control_train])*prct_test*prct_struct_control )
Index_heme_test = sample(1:length(names_heme_list[-Index_heme_train]),size = length(names_heme_list[-Index_heme_train])*prct_test*prct_struct_heme )
Index_nucleotide_test = sample(1:length(names_nucleotide_list[-Index_nucleotide_train]),size = length(names_nucleotide_list[-Index_nucleotide_train])*prct_test*prct_struct_nucleotide )
Index_steroid_test = sample(1:length(names_steroid_list[-Index_steroid_train]),size = length(names_steroid_list[-Index_steroid_train])*1*prct_struct_steroid ) #set to 1 for all steroid data in tests
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

names = c(names_control_train,names_heme_train,names_nucleotide_train,names_steroid_train)
categories = c(rep(0,length(names_control_train)), rep(1,length(names_heme_train)), rep(2,length(names_nucleotide_train)))

y_train = to_categorical(categories)
for (i in 1:n_struct_train) {
  x_train[i, , , , ] = np$load(paste0(paste('../data/deepdrug3d_voxel_data/', names[i],sep = ""),".npy"))
}
####x_test et y_test####
x_test = array(data = NA, dim = c(n_struct_test, 14, 32, 32, 32))
names_test = c(names_control_test,names_heme_test,names_nucleotide_test,names_steroid_test)
categories = c(rep(0,length(names_control_test)), rep(1,length(names_heme_test)), rep(2,length(names_nucleotide_test)), rep(3,length(names_steroid_test)))
y_test = to_categorical(categories)
for (i in 1:n_struct_test) {
  x_test[i, , , , ] = np$load(paste0(paste('../data/deepdrug3d_voxel_data/', names_test[i],sep = ""), ".npy"))
}
#### MODEL ####
#attention à mettre - data_format='channels_first' - partout
#model 1 - model article
model_1 = function(){
  model = keras_model_sequential()
  layer_conv_3d(model, filters = 64 , kernel_size = 5, activation = "relu", input_shape = c(14,32,32,32), data_format='channels_first')
  layer_conv_3d(model, filters = 64 , kernel_size = 3, activation = "relu", data_format='channels_first')
  layer_dropout(model, 0.2)
  layer_max_pooling_3d(model, pool_size = c(2,2,2), data_format='channels_first')
  layer_dropout(model, 0.4)
  
  layer_flatten(model)
  
  layer_dense(model, units = 128, activation = "relu")
  layer_dropout(model, 0.4)
  layer_dense(model, units = 3, activation = "softmax")
  return(model)
}

#
#modele 2 perso :
model_2 = function(){
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
}

#
#modele 3 resnet
model_3 = function(){
  input = layer_input(shape = c(14,32,32,32))
  output = layer_conv_3d(input, filters = 8, kernel_size = c(3,3,3), padding = "same", 
                             activation = "relu", kernel_initializer = "he_normal", data_format='channels_first')
  for(i in 1:2){
    save = output
    if(save$shape[-1] != 8){
      output = layer_conv_3d(output, filters = 64, kernel_size = c(3,3,3), padding = "same", 
                             activation = "relu", kernel_initializer = "he_normal", data_format='channels_first')
    }
    output1 = layer_conv_3d(output, filters = 32, kernel_size = c(3,3,3), activation = "relu",
                            padding = "same", kernel_initializer = "he_normal", data_format='channels_first')
    output2 = layer_conv_3d(output1, filters = 64, kernel_size = c(3,3,3), activation = "linear",
                            padding = "same", kernel_initializer = "he_normal", data_format='channels_first')
    conc = layer_add(inputs = list(output2, save))
    activ = layer_activation(conc, activation = "relu")
    output = activ
  }
  mod = output
  
  model_max1 = layer_max_pooling_3d(mod, pool_size = c(3,3,3), data_format='channels_first') 
  model_drop1 = layer_dropout(model_max1, 0.4)
  model_flatten1 = layer_flatten(model_drop1)
  model_dense1 = layer_dense(model_flatten1, units = 3, activation = "softmax")
  
  model = keras_model(inputs = input, outputs = model_dense1)
  return(model)
}

#
model_train = function(model, epochs, x_train, y_train){
  compile(model,loss = 'categorical_crossentropy',optimizer = optimizer_adam(), metrics = "accuracy")
  history = fit(model, x_train, y_train, epochs = epochs, batch_size = 32, validation_split = 0.3)
  return(model)
}


#### Performances sur test####
#
model_accuracy = function(model, x_test, y_test){
  print("Accuracy du modèle sur test")
  print(evaluate(model, x_test, y_test))
  model_predict_test  = predict(model, x_test)
  print("predict")
  model_predict_test_val = NULL
  model_predict_test_val = max.col(model_predict_test)
  pred = factor(model_predict_test_val, 1:3)
  test = factor(max.col(y_test), 1:3)
  TableTot = table(pred, test)
  TP = TableTot[2,2] + TableTot[3,3]
  FP = TableTot[2,1] + TableTot[3,1] + TableTot[1,2] + TableTot[1,3]
  FN = TableTot[2,3] + TableTot[3,2]
  TN = TableTot[1,1]
  
  ACC = (TP+TN)/(FP+FN+TP+TN)
  PPV = TP/(TP+FP)
  TNR = TN/(TN+FP)
  TPR = TP/(TP+FN)
  FPR = TP/(FP+TN)
  
  print("Table")
  print(TableTot)
  print("ACC")
  print(ACC)
  print("PPV")
  print(PPV)
  print("TNR")
  print(TNR)
  print("TPR")
  print(TPR)
  print("FPR")
  print(FPR)
  #accuracy = sum(diag(TableTot))/sum(TableTot)
  #print(accuracy)
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
  dt.pred = prediction(model_predict_test, y_test)
  dt.perf = performance(dt.pred, "tpr", "fpr")
  
  plot(dt.perf)
  title("Courbe Roc des modèle en Validation")
  dt.auc = performance(dt.pred, "auc")
  abline(0,1)
  
  print(attr(dt.auc, "y.values"))
}
#### Tests ####
epochs = 20
#accuracy on control negatif steroid
x_test_steroid  = model_negativ_test(model3, names_steroid_list)
categories = c(rep(0,length(names_steroid_list)))
y_test_steroid = to_categorical(categories,3)
#modele1
model1 = model_1()
model1 = model_train(model1, epochs, x_train, y_train)
sink("../results/model_test_model1.txt")
model_accuracy(model1, x_test, y_test)
print("test on steroid")
model_accuracy(model1, x_test_steroid, y_test_steroid)
jpeg("../results/plot_roc_model1.jpeg") 
model_ROC_curve(model1, x_test, y_test)
dev.off()
sink()
save_model_hdf5(model1, "../results/model_test_model1.h5")

#modele2
model2 = model_2()
model2 = model_train(model2, epochs, x_train, y_train)
sink("../results/model_test_model2.txt")
model_accuracy(model2, x_test, y_test)
print("test on steroid")
model_accuracy(model1, x_test_steroid, y_test_steroid)
jpeg("../results/plot_roc_model2.jpeg") 
model_ROC_curve(model2, x_test, y_test)
dev.off()
sink()
save_model_hdf5(model2, "../results/model_test_model2.h5")

#modele2
model3 = model_3()
model3 = model_train(model3, epochs, x_train, y_train)
sink("../results/model_test_model3.txt")
model_accuracy(model3, x_test, y_test)
print("test on steroid")
model_accuracy(model1, x_test_steroid, y_test_steroid)
jpeg("../results/plot_roc_model3.jpeg") 
model_ROC_curve(model3, x_test, y_test)
dev.off()
sink()
save_model_hdf5(model3, "../results/model_test_model3.h5")

####save model####

####test vrai model####
#model = load_model_hdf5("deepdrug3d.h5")







