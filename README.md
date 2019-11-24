# EIP4
Learning DNN

#Assignment-2
    
#copy and paste your Logs for 20 epochs
    Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.002.
60000/60000 [==============================] - 40s 662us/step - loss: 0.0897 - acc: 0.9570 - val_loss: 0.0219 - val_acc: 0.9930
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0015163002.
60000/60000 [==============================] - 9s 146us/step - loss: 0.0864 - acc: 0.9575 - val_loss: 0.0206 - val_acc: 0.9945
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0012210012.
60000/60000 [==============================] - 9s 142us/step - loss: 0.0828 - acc: 0.9593 - val_loss: 0.0195 - val_acc: 0.9942
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0010219724.
60000/60000 [==============================] - 9s 143us/step - loss: 0.0818 - acc: 0.9589 - val_loss: 0.0229 - val_acc: 0.9942
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0008787346.
60000/60000 [==============================] - 9s 144us/step - loss: 0.0800 - acc: 0.9592 - val_loss: 0.0175 - val_acc: 0.9946
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0007707129.
60000/60000 [==============================] - 9s 145us/step - loss: 0.0806 - acc: 0.9588 - val_loss: 0.0214 - val_acc: 0.9952
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0006863418.
60000/60000 [==============================] - 9s 143us/step - loss: 0.0777 - acc: 0.9601 - val_loss: 0.0199 - val_acc: 0.9950
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0006186205.
60000/60000 [==============================] - 9s 144us/step - loss: 0.0775 - acc: 0.9600 - val_loss: 0.0224 - val_acc: 0.9941
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0005630631.
60000/60000 [==============================] - 9s 145us/step - loss: 0.0765 - acc: 0.9595 - val_loss: 0.0193 - val_acc: 0.9950
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0005166624.
60000/60000 [==============================] - 9s 144us/step - loss: 0.0766 - acc: 0.9611 - val_loss: 0.0174 - val_acc: 0.9950
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.000477327.
60000/60000 [==============================] - 9s 143us/step - loss: 0.0766 - acc: 0.9602 - val_loss: 0.0169 - val_acc: 0.9953
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.0004435573.
60000/60000 [==============================] - 9s 144us/step - loss: 0.0759 - acc: 0.9610 - val_loss: 0.0182 - val_acc: 0.9947
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0004142502.
60000/60000 [==============================] - 9s 143us/step - loss: 0.0755 - acc: 0.9601 - val_loss: 0.0174 - val_acc: 0.9950
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0003885759.
60000/60000 [==============================] - 9s 149us/step - loss: 0.0781 - acc: 0.9584 - val_loss: 0.0174 - val_acc: 0.9947
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0003658983.
60000/60000 [==============================] - 9s 144us/step - loss: 0.0757 - acc: 0.9604 - val_loss: 0.0157 - val_acc: 0.9953
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0003457217.
60000/60000 [==============================] - 9s 146us/step - loss: 0.0755 - acc: 0.9599 - val_loss: 0.0169 - val_acc: 0.9956
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000327654.
60000/60000 [==============================] - 9s 142us/step - loss: 0.0734 - acc: 0.9606 - val_loss: 0.0170 - val_acc: 0.9947
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.000311381.
60000/60000 [==============================] - 9s 145us/step - loss: 0.0739 - acc: 0.9604 - val_loss: 0.0163 - val_acc: 0.9953
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0002966479.
60000/60000 [==============================] - 9s 143us/step - loss: 0.0733 - acc: 0.9606 - val_loss: 0.0172 - val_acc: 0.9953
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000283246.
60000/60000 [==============================] - 9s 144us/step - loss: 0.0725 - acc: 0.9616 - val_loss: 0.0171 - val_acc: 0.9959
    
    
    
#copy and paste the result of your model.evaluate (on test data)
[0.017078954667203654, 0.9959]

#Strategy you have taken to achieve the said results

The following strategy was adopted
1. a Receptive field of 5x5 was used to ensure that the data was learned in sufficient detail before going through MaxPooling
   - For this 2 convolutions with 16 layers each was used as 32 layers was increasing the number of parameters
2. After this 1x1 convolution is used to extract features unto 10 layers.
3. Maxpooling reduces the number of parameters to keep it below 15k
4. after this 4 sequential 16 layer 3x3 convolution is used to learn the features.
5. This keeps the total number of parameters to close to 14k
6. A learning rate of 0.002 was used to ensure that accuracy was achieved quickly and properly.

The above strategy led to the best result.



