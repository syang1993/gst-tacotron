import argparse
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


sentences = [
  # From July 8, 2017 New York Times:
  'Scientists at the CERN laboratory say they have discovered a new particle.',
  'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
  'President Trump met with other leaders at the Group of 20 conference.',
  'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
  # From Google's Tacotron example page:
  'Generative adversarial network or variational auto-encoder.',
  'The buses aren\'t the problem, they actually provide a solution.',
  'Does the quick brown fox jump over the lazy dog?',
  'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  is_teacher_force = False
  mel_targets = args.mel_targets
  if args.mel_targets is not None:
    is_teacher_force = True
    mel_targets = np.load(args.mel_targets)
  synth = Synthesizer(teacher_forcing_generating=is_teacher_force)
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    if args.text is not None:
      path = '%s-eval.wav' % (base_path)
      print('Synthesizing: %s' % path)
      reference_mel = args.reference_mel
      if reference_mel is not None:
        reference_mel = np.load(args.reference_mel)
      else:
        print("TODO: add style weights when there is no reference mel")
        raise
      with open(path, 'wb') as f:
        f.write(synth.synthesize(args.text, mel_targets=mel_targets, reference_mel=reference_mel))
      break
    else:
      path = '%s-%d.wav' % (base_path, i)
      print('Synthesizing: %s' % path)
      with open(path, 'wb') as f:
        f.write(synth.synthesize(text))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--text', default=None, help='Single test text sentence')
  parser.add_argument('--reference_mel', default=None, help='Reference mel path')
  parser.add_argument('--mel_targets', default=None, help='Mel-targets path, used when use teacher_force generation')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
