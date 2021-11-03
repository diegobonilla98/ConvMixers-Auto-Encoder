# ConvMixers-Auto-Encoder
An idea for a generator network using the new ConvMixer's architecture. 

As the ConvMixer introduces a convolution layer to do the patches, this autoencoder just consists on a Transpose Convolution undoing the first patching convolution (followed by an UpSampling + Conv layer to smooth the patchy results).


## Results
Only trained for 10 epochs on the full [https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets/](IA Segment Human Matting Dataset).

![](./pantallazo.png)
