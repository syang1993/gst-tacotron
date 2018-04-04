import argparse
import os
import re
import numpy as np
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer

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
  synth.load(args.checkpoint, args.reference_mel)
  base_path = get_output_base_path(args.checkpoint)

  path = '%s-eval.wav' % (base_path)
  print('Synthesizing: %s' % path)
  reference_mel = args.reference_mel
  if reference_mel is not None:
    reference_mel = np.load(args.reference_mel)
  else:
    if hp.use_gst:
      #raise ValueError("TODO: add style weights when there is no reference mel. Now we use random weights.")
      reference_mel=None
    else:
      raise ValueError("You must set the reference audio if you don't want to use GSTs.")

  with open(path, 'wb') as f:
    f.write(synth.synthesize(args.text, mel_targets=mel_targets, reference_mel=reference_mel))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--text', required=True, default=None, help='Single test text sentence')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--reference_mel', default=None, help='Reference mel path')
  parser.add_argument('--mel_targets', default=None, help='Mel-targets path, used when use teacher_force generation')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
