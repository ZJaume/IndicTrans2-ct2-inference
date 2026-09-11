# IndicTrans2-ct2-inference
Fork of [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) that removes all the bloat and provides easy to use Ctranslate2 inference.

## Installation
```
pip install git+https://github.com/ZJaume/IndicTrans2-ct2-inference.git
```

## Usage
The package downloads `indictrans2-1B` model from HuggingFace, so it is required to be [logged in](https://huggingface.co/docs/huggingface_hub/quick-start) before usinc CLI or Python package.

### Python Package
```python
from indictrans2_ct2_inference.translate import Translator
t = Translator(src_lang='eng_Latn', trg_lang='ben_Beng', device_index=[0], mini_batch_size=2000)
t.batch_translate(['hello','good bye'])
```

### CLI
```
usage: indictrans2-inference [-h] [--gpus GPUS] [--nbest] [-i INPUT] [-o OUTPUT] [-M MAXI_BATCH] [-m MINI_BATCH]
                             [-b BEAM_SIZE]
                             src_lang trg_lang

positional arguments:
  src_lang
  trg_lang

options:
  -h, --help            show this help message and exit
  --gpus GPUS
  --nbest               Produce fake nbest format
  -i INPUT, --input INPUT
                        Instead of stdin, read from this file
  -o OUTPUT, --output OUTPUT
                        Instead of stdout, wirte to this file
  -M MAXI_BATCH, --maxi_batch MAXI_BATCH
  -m MINI_BATCH, --mini_batch MINI_BATCH
  -b BEAM_SIZE, --beam_size BEAM_SIZE
```
