# Convert StableHLO models into Apple Core ML format

* This repo is currently experimental! *

## Converting a model

To convert a StableHLO module, do the following:

```python
import coremltools as ct
from stablehlo_coreml.converter import convert

mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
cml_model = ct.convert(mil_program, source="milinternal", minimum_deployment_target=ct.target.iOS18)
```

## Notes
* `coremltools` supports up to python 3.11. Do not run hatch with a newer version.
  Can be controlled using fx `export HATCH_PYTHON=python3.9`
* Run tests using `hatch test -i py=3.9`
  * Currently an error is happening, where the `so` files does not make it into the `hatch-test.py3.9` environment. I need to figure out why.
    I have currently copied in the files manually...
