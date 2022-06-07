# AmplitudeCrafter
## Aim
This Library aims to automize the creation of amplitudes using the jitter fitting framework. Functions used as resonances inside the amplitudes may also stem from different libraries, but have to be jit-able. 

## Usage
The main usecase is the DalitzAmplitude class. Here a Dalitz decay can be specified over one or mutiple yml files. 
An example with detailed commentary is provided in the config folder of this library and can be loaded via the inbuild path at
```AmplitudeCrafter.locals.decay_example```

## Features
- particle class with predefied particles for simpler amplitude construction (work in progress on adding all particles)
- resonance class encapsuleing any lineshape
- Dalitz Amplitude Constructor returning a fittable function and the ability to read and dump yml files
