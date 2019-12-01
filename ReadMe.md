# EIP4
Learning DNN

# Assignment 3

Base Network - Final Validation accuracy = 82.69%
Model created - Best validation accuracy = 85.94%

# Model Definition

dp=0.1

model = Sequential()
 
model.add(SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,3), depth_multiplier = 16)) ##Output size = 30x30 #Receptive field = 3x3 
model.add(BatchNormalization())
model.add(Dropout(dp))

model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', depth_multiplier = 2)) ##Output size = 28x28 #Receptive field = 5x5
model.add(BatchNormalization())
model.add(Dropout(dp))

model.add(MaxPooling2D(pool_size=(2, 2))) #output size 14x14 

model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', depth_multiplier = 2)) ##Output size = 12x12 #Receptive field after maxpooling = 3x3
model.add(BatchNormalization())
model.add(Dropout(dp))

model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', depth_multiplier = 2)) ##Output size = 10x10 #Receptive field after maxpooling = 5x5
model.add(BatchNormalization())
model.add(Dropout(dp))

model.add(MaxPooling2D(pool_size=(2, 2)))#output size 5x5

model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', depth_multiplier = 2)) ##Output size = 3x3 #Receptive field after maxpooling = 3x3
model.add(BatchNormalization())
model.add(Dropout(dp))

model.add(SeparableConv2D(filters=10, kernel_size=(3, 3), activation='relu', depth_multiplier = 1))  ##Output size = 1x1 #Receptive field = 5x5
model.add(Flatten())
model.add(Activation('softmax'))

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 50 epoch logs

Epoch 1/50
585/585 [==============================] - 123s 211ms/step - loss: 1.0904 - acc: 0.6166 - val_loss: 1.0586 - val_acc: 0.6479
Epoch 2/50
585/585 [==============================] - 123s 209ms/step - loss: 0.8481 - acc: 0.7040 - val_loss: 0.8476 - val_acc: 0.7125
Epoch 3/50
585/585 [==============================] - 122s 209ms/step - loss: 0.7471 - acc: 0.7387 - val_loss: 0.7321 - val_acc: 0.7526
Epoch 4/50
585/585 [==============================] - 123s 210ms/step - loss: 0.6809 - acc: 0.7632 - val_loss: 0.7229 - val_acc: 0.7590
Epoch 5/50
585/585 [==============================] - 122s 209ms/step - loss: 0.6366 - acc: 0.7779 - val_loss: 0.5847 - val_acc: 0.8023
Epoch 6/50
585/585 [==============================] - 122s 209ms/step - loss: 0.6022 - acc: 0.7895 - val_loss: 0.6201 - val_acc: 0.7957
Epoch 7/50
585/585 [==============================] - 122s 209ms/step - loss: 0.5718 - acc: 0.8002 - val_loss: 0.5485 - val_acc: 0.8147
Epoch 8/50
585/585 [==============================] - 122s 209ms/step - loss: 0.5497 - acc: 0.8083 - val_loss: 0.5899 - val_acc: 0.8033
Epoch 9/50
585/585 [==============================] - 122s 209ms/step - loss: 0.5283 - acc: 0.8157 - val_loss: 0.6053 - val_acc: 0.8005
Epoch 10/50
585/585 [==============================] - 122s 209ms/step - loss: 0.5115 - acc: 0.8217 - val_loss: 0.5991 - val_acc: 0.8006
Epoch 11/50
585/585 [==============================] - 122s 209ms/step - loss: 0.4940 - acc: 0.8275 - val_loss: 0.5352 - val_acc: 0.8241
Epoch 12/50
585/585 [==============================] - 122s 208ms/step - loss: 0.4815 - acc: 0.8317 - val_loss: 0.5524 - val_acc: 0.8205
Epoch 13/50
585/585 [==============================] - 122s 208ms/step - loss: 0.4676 - acc: 0.8367 - val_loss: 0.5498 - val_acc: 0.8195
Epoch 14/50
585/585 [==============================] - 122s 208ms/step - loss: 0.4607 - acc: 0.8387 - val_loss: 0.4850 - val_acc: 0.8372
Epoch 15/50
585/585 [==============================] - 121s 207ms/step - loss: 0.4504 - acc: 0.8426 - val_loss: 0.5114 - val_acc: 0.8273
Epoch 16/50
585/585 [==============================] - 121s 207ms/step - loss: 0.4420 - acc: 0.8456 - val_loss: 0.5158 - val_acc: 0.8316
Epoch 17/50
585/585 [==============================] - 121s 208ms/step - loss: 0.4287 - acc: 0.8489 - val_loss: 0.5087 - val_acc: 0.8364
Epoch 18/50
585/585 [==============================] - 121s 207ms/step - loss: 0.4217 - acc: 0.8518 - val_loss: 0.5822 - val_acc: 0.8193
Epoch 19/50
585/585 [==============================] - 121s 207ms/step - loss: 0.4126 - acc: 0.8555 - val_loss: 0.5285 - val_acc: 0.8260
Epoch 20/50
585/585 [==============================] - 121s 207ms/step - loss: 0.4085 - acc: 0.8564 - val_loss: 0.5268 - val_acc: 0.8319
Epoch 21/50
585/585 [==============================] - 121s 207ms/step - loss: 0.4028 - acc: 0.8587 - val_loss: 0.5409 - val_acc: 0.8193
Epoch 22/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3932 - acc: 0.8617 - val_loss: 0.4806 - val_acc: 0.8363
Epoch 23/50
585/585 [==============================] - 121s 208ms/step - loss: 0.3904 - acc: 0.8624 - val_loss: 0.4478 - val_acc: 0.8521
Epoch 24/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3835 - acc: 0.8653 - val_loss: 0.5091 - val_acc: 0.8378
Epoch 25/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3802 - acc: 0.8665 - val_loss: 0.4694 - val_acc: 0.8468
Epoch 26/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3710 - acc: 0.8692 - val_loss: 0.5604 - val_acc: 0.8257
Epoch 27/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3641 - acc: 0.8726 - val_loss: 0.4784 - val_acc: 0.8496
Epoch 28/50
585/585 [==============================] - 122s 208ms/step - loss: 0.3627 - acc: 0.8720 - val_loss: 0.4526 - val_acc: 0.8516
Epoch 29/50
585/585 [==============================] - 121s 208ms/step - loss: 0.3608 - acc: 0.8733 - val_loss: 0.5318 - val_acc: 0.8328
Epoch 30/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3545 - acc: 0.8754 - val_loss: 0.4824 - val_acc: 0.8426
Epoch 31/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3519 - acc: 0.8760 - val_loss: 0.4598 - val_acc: 0.8503
Epoch 32/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3464 - acc: 0.8779 - val_loss: 0.5151 - val_acc: 0.8338
Epoch 33/50
585/585 [==============================] - 121s 208ms/step - loss: 0.3448 - acc: 0.8788 - val_loss: 0.4534 - val_acc: 0.8486
Epoch 34/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3389 - acc: 0.8801 - val_loss: 0.4329 - val_acc: 0.8575
Epoch 35/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3356 - acc: 0.8821 - val_loss: 0.5107 - val_acc: 0.8420
Epoch 36/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3330 - acc: 0.8827 - val_loss: 0.4869 - val_acc: 0.8456
Epoch 37/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3286 - acc: 0.8842 - val_loss: 0.4860 - val_acc: 0.8472
Epoch 38/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3295 - acc: 0.8840 - val_loss: 0.5166 - val_acc: 0.8340
Epoch 39/50
585/585 [==============================] - 121s 208ms/step - loss: 0.3247 - acc: 0.8855 - val_loss: 0.4305 - val_acc: 0.8594
Epoch 40/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3232 - acc: 0.8860 - val_loss: 0.4696 - val_acc: 0.8488
Epoch 41/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3184 - acc: 0.8878 - val_loss: 0.5491 - val_acc: 0.8302
Epoch 42/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3158 - acc: 0.8885 - val_loss: 0.4651 - val_acc: 0.8515
Epoch 43/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3167 - acc: 0.8881 - val_loss: 0.4266 - val_acc: 0.8572
Epoch 44/50
585/585 [==============================] - 121s 208ms/step - loss: 0.3121 - acc: 0.8893 - val_loss: 0.5039 - val_acc: 0.8392
Epoch 45/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3130 - acc: 0.8894 - val_loss: 0.4201 - val_acc: 0.8573
Epoch 46/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3110 - acc: 0.8901 - val_loss: 0.4498 - val_acc: 0.8545
Epoch 47/50
585/585 [==============================] - 122s 208ms/step - loss: 0.3045 - acc: 0.8921 - val_loss: 0.4761 - val_acc: 0.8476
Epoch 48/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3054 - acc: 0.8921 - val_loss: 0.4910 - val_acc: 0.8447
Epoch 49/50
585/585 [==============================] - 121s 208ms/step - loss: 0.3025 - acc: 0.8935 - val_loss: 0.5197 - val_acc: 0.8394
Epoch 50/50
585/585 [==============================] - 121s 207ms/step - loss: 0.3007 - acc: 0.8938 - val_loss: 0.4949 - val_acc: 0.8465
Model took 6078.38 seconds to train

// Accuracy on test data is: 84.65
// Best validation accuracy = 85.94%
// Epoch 39/50 - val_acc: 0.8594
