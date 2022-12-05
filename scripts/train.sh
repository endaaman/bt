epoch=200

# python train.py train -e $epoch -m eff_v2_b0 -b 24
# python train.py train -e $epoch -m eff_v2_b1 -b 16
python train.py train -e $epoch -m eff_v2_b2 -b 16
python train.py train -e $epoch -m eff_v2_b3 -b 16
python train.py train -e $epoch -m eff_b7_ns -b 2
python train.py train -e $epoch -m eff_b6_ns -b 4
python train.py train -e $epoch -m eff_b5_ns -b 4
python train.py train -e $epoch -m eff_b4_ns -b 4
python train.py train -e $epoch -m eff_b3_ns -b 8
python train.py train -e $epoch -m eff_b2_ns -b 8
python train.py train -e $epoch -m eff_b1_ns -b 8
python train.py train -e $epoch -m eff_b0_ns -b 16
