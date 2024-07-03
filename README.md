# Malignant brain tumor classification

## Experiment conditions

```
$ python compare.py train --base <BASE> --encoder <ENCODER> --limit <LIMIT>
```

- `BASE`
    - `random`
    - `imagenet`
    - `uni`
- `ENCODER`
    - `frozen`
    - `unfrozen`
- `LIMIT`
    - `10`
    - `100`
    - `500`

```
$ parallel pueue add python compare.py train --base {1} --encoder frozen --code {2} ::: uni imagenet random ::: LMGGGB LMGAOB
```

#### frozen

```
$ parallel pueue add python compare.py train --base {1} --encoder frozen --code {2} ::: uni imagenet random ::: LMGGGB LMGAOB
```

#### unfrozen

```
$ parallel pueue add python compare.py train --encoder unfrozen --epoch 10 --base {2} --code {1} ::: LMGGGB LMGAOB ::: uni imagenet random
```
