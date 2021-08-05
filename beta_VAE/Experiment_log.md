# Model Structure:

## 1st Generation:
-  Latent Dimension: MNIST 8  , Celeb 64 
- Encoder: 2 Conv2D(layers 3 by 3 filter), decoder: 2 Conv2D (2,2) filter.
```python
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=shape),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=self.output_f * self.output_f *32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(self.output_f, self.output_f, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=self.output_s, kernel_size=3, strides=1, padding='same'),
        ]
    )

```

## 2nd Generation (3/17/21)
-  Latent Dimension: MNIST 8  , Celeb 64 
-  3 conv2d encoder, 3 conv2d layers decoder

## 5/17 model (s_decoder train to encode angle and apply the back propogate )
- MNIST test 7 full range digit 7, partial range digit 9;
- 5/21 mode test 8 full range digit 7, partial range digit 3; 
- 5/26 mode test9 full range digit 7, partial range digit 3 with training method(futher_dis) setting, ;
- 5/31: 
    - mnist_test 12 full range digit [4,5,6], partial range digit 3;
    - teacher_network full range digit 7,
- 6/6:
    - training data:
        -   full range: 7 [:100], parital range:9 [:100] 
    - teacher_network1, without bias vector (both decoder and encoder)
    - teacher_network2, without bias (decoder) 
    - student_network1, without bias (decoder)
    
- mnist_test14 (6/6):
    - beta_tcvae model
    - full range digit 4, partial range:3
- mnist_test15 (6_8)
    - with shuffle 
    
- mnist_test16 (6_8)
    - full range (line); partial range 1 (*)
- mnist test17
    - full range (line); partial range 3 (*)
    
- 6/9:
    - teacher_network3, line angle (0-360)
    - student_network3, 3 (0-180)
    - student_network4, 3 (0)
    - merge_network2, combine teacher network3 and student_network3 (*)
    - merge_network3, combine teacher network3 and student_network4 (*)
    
- 6/10:
    - mnist_test18 full range (line) 0-360; partial range(3) 0;
    - mnist_test19 full range (line) 0-360, partial range[3, 4, 5], (0-180)
    - mnist_test20 full range (line, 10,000 sample), partial range (3) (0-180)
- 6/12
    - mnist_test21 full range(line, 1,000 sample), partial range(3, 0-180)
        - training procedure: 1) train full range set individually; then, mix full range and partial range.
    - width_test1  full range(line, 1,000 sample, width 0-3) partial range (1, 0)

-6/13
    - dsprite test1
    
    ```         full_index = latents_classes.loc[((latents_classes['shape'] == 2) &
                                          (latents_classes['x_axis'] == 15) &
                                          (latents_classes['y_axis'] == 15))].index
        full.append(full_index)
        partial_index = latents_classes.loc[((latents_classes['shape'] == 1) &
                                                 (latents_classes['x_axis'] == 15) &
                                                 (latents_classes['y_axis'] == 15))].index
    ```
    
    - dsprite test2 reverse training process
    
    - dsprite test3 (reverse training process) change shape (full range 1, partial range 0)
    
    - dsprite test 4 shape (full range 1, partial range 0)
    
    - dsprite test 5 shape (full range 2, partial range 0)
    
- 6/20
    - simclr test3:
    
    full range set (0-360) digit 3, partial (3,4), 30 epochs
    
    - baseline_model: 80%
    
    - simclr test4:
    full range set (0-360) digit 3, partial all digit(accuracy_rate=71%)

- 6/27
    - simclr test 5:
    full range set (0-360) digit 0, partial range all digit (accuracy rate=70%) 
    
    - baseline_model: full_range 0, partial all digits (acc=76.84%)
    
    - simclr test 6:
    full range set (0-360) digit 0, partial range all digit (accuracy rate) 
    classifier also learn the rotated z for oversample (acc=72.6%).
    
   - simclr test 7:
   oversampling (from 0, 360 degree for all data, acc=95.97%) without adjust decoder;
   
   - simclr test8:
   oversampling (from 0 to 360 degree, acc=86%) adjust decoder; 
   
- 7/8
    - baseline classifier(mlp acc:80%; cnn acc:90%)
    
    - mnist test2:
        loss function: classifier use decode data from generator for optimization;
        
    - mnist test3 (elbo:105 acc:81%):
        loss funciton: classifier and generator train seperatly; classifier will train first;
        
    - mnist test4 (elbo:102, acc: 87%):
        same loss function with 3 but used oversampling;
        
    - mnist test 6 (elbo:87, acc:80%)
        classifier and generator used cnn 
        
    - mnist test 7 (elbo:, acc)
        classifier used cnn
        
- 7/13
    - mnist test 8 (elbo:,acc)
        adjusting classifier first;
        
        
- 7/15 
    - mnist test 10(elbo, acc)
     oversampling will loop and apply the classifier loss to generator, gamma=4;
     
   - mnist test 11 (gm=91.32, asca=91.84)
     oversampling will loop and apply the classifier loss to generator,gamma = 1

- 7/18
   - mnist test 12 (gm, asca)
        oversampling apply confidence estimation; gamma=2
        
   - mnist test 15 
        with misample loss
        
- 7/20 
    - cifar10 baseline (acc=0.36)
    
- 7/21 (consevative, margin, 0.85,)
    - mnist20 (g_means: 0.96, acsa:0.968)
    - fashion mnist (g_mears:0.8, asca:0.81)

- 8/4
    - celebA test 7 (margin, 1)
        - acsa: o_g_means:0.7494,  o_acsa:0.620