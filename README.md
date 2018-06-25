# GST Tacotron (exprissive end-to-end speech syntheis using global style token)

A tensorflow implementation of the [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017) and [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047).


### Audio Samples

  * **[Audio Samples](https:///syang1993.github.io/gst-tacotron/)** from models trained using this repo with default hyper-params.
    * This set was trained using the [Blizzard 2013 dataset](https://www.synsig.org/index.php/Blizzard_Challenge_2013) with and without global style tokens (GSTs).
      * I found the synthesized audio can learn the prosody of the reference audio.
      * The audio quality isn't so good as the paper. Maybe more data, more training steps and the wavenet vocoder will improve the quality, as well as better attention mechanism.
      

## Quick Start:

### Installing dependencies

1. Install Python 3.

2. Install the latest version of [TensorFlow](https://www.tensorflow.org/install/) for your platform. For better performance, install with GPU support if it's available. This code works with TensorFlow 1.4.

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```

### Training

1. **Download a dataset:**

   The following are supported out of the box:
    * [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) (Public Domain)
    * [Blizzard 2013](https://www.synsig.org/index.php/Blizzard_Challenge_2013) (Creative Commons Attribution Share-Alike)

   We use the Blizzard 2013 dataset to test this repo (Google's paper used 147 hours data read by the 2013 Blizzard Challenge speaker). This year Challenge provides about 200 hours unsegmented speech and 9741 segmented waveforms, I did all the experiments based the 9741 segmented waveforms since it's hard for me to split the unsegmented data.

   You can use other datasets if you convert them to the right format. See more details about data pre-process in keithito's [TRAINING_DATA.md](https://github.com/keithito/tacotron/blob/master/TRAINING_DATA.md).

2. **Preprocess the data**
    
   ```
   python3 preprocess.py --dataset blizzard2013
   ```

3. **Train a model**

   ```
   python3 train.py
   ```
   
   The above command line will use default hyperparameters, which will train a model with cmudict-based phoneme sequence and 4-head multi-head sytle attention for global style tokens. If you set the `use_gst=False` in the hparams, it will train a model like Google's another paper [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047).

   Tunable hyperparameters are found in [hparams.py](hparams.py). You can adjust these at the command line using the `--hparams` flag, for example `--hparams="batch_size=16,outputs_per_step=2"` . Hyperparameters should generally be set to the same values at both training and eval time.

4. **Synthesize from a checkpoint**

   ```
   python3 eval.py --checkpoint ~/tacotron/logs-tacotron/model.ckpt-185000 --text "hello text" --reference_audio /path/to/ref_audio
   ```

    Replace "185000" with the checkpoint number that you want to use. Then this command line will synthesize a waveform with the content "hello text" and the style of the reference audio. If you don't use the `--reference_audio`, it will generate audio with random style weights, which may generate unintelligible audio sometimes. 

   If you set the `--hparams` flag when training, set the same value here.


## Notes:

  Since the paper didn't talk about the details of the style-attention layer, I'm a little confused about the global style tokens. For the token embedding (GSTs) size, the paper said that they set the size to 256/h, where `h` is the number of heads. I'm not sure whether I should initialize the same or different GSTs as attention memory for all heads.

## Reference
  -  Keithito's implementation of tacotron: https://github.com/keithito/tacotron
  -  Yuxuan Wang, Daisy Stanton, Yu Zhang, RJ Skerry-Ryan, Eric Battenberg, Joel Shor, Ying Xiao, Fei Ren, Ye Jia, Rif A. Saurous. 2018. [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017)
  - RJ Skerry-Ryan, Eric Battenberg, Ying Xiao, Yuxuan Wang, Daisy Stanton, Joel Shor, Ron J. Weiss, Rob Clark, Rif A. Saurous. 2018. [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047).
