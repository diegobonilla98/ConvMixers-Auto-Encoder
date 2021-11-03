# ConvMixers-Auto-Encoder
An idea for a generator network using the new ConvMixer's architecture. 

As the ConvMixer introduces a convolution layer to do the patches, this autoencoder just consists on a Transpose Convolution undoing the first patching convolution (followed by an UpSampling + Conv layer to smooth the patchy results).
