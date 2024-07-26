# Fuyu
This repo is an attempt to replicate the Multimodal [Fuyu](https://www.adept.ai/blog/fuyu-8b) architecture proposed by Adept AI.

Multimodal architectures are meant to process text, images and audio, and are incredibly useful for AI agents trying to complete tasks in the "real world". 
The particularity of the Fuyu architecture is that images are patched and processed as text (bypassing the first embedding dimension) by a decoder, meaning a simple vanilla decoder-only Transformer is needed, instead of having separated encoder modules to process the non-text inputs.

In general, the simplicity of the architecture is remarkable, and provides more lighter and easier to scale models relative to its counterparts.

## File Structure
The modules are organized in files depending on their purpose. An [experiment](fuyu.ipynb) notebook is also added in case someone wants to play with the model and tinker with the parameters and/or implementation.

Also a [test](test.py) file is provided to test the forward passes on the architecture, both for passes with targets and without targets, as well as for generation.

```
python test.py
```

## Notes
- I have doubts on wether the image transformation to text has to be learnable. My gut tells me no, but I provided both implementations just in case.
- This code is meant more as an exercise and not optimized for training.
- I have some doubts regarding the implementation of the qv values normalization during Attention.
