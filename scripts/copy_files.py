import os
import re
import time
import filecmp
import shutil
from glob import glob
from tqdm import tqdm

from pydantic import Field
from endaaman.cli import BaseCLI

targets = {
    'L': [
        'N21-231',
        'N21-297',
        'N21-306',

        'N22-040',
        'N22-044',
        'N22-063',
        'N22-079',
        'N22-100',
        'N22-113',
        'N22-184',
        'N22-188',
        'N22-231',
        'N22-255',
        'N22-263',
        'N22-277',
        'N22-295',
    ],

    'A': [
        'N20-138',
        'N20-158',
        'N20-276',
        'N22-108',
        'N22-163',
        'N22-166',
        'N22-193',
        'N22-207',
    ],

    'O': [
        # FEW CASES
    ],

    'G': [
        'N21-018',
        'N21-051',
        'N21-081',
        'N21-153',
        'N21-181',
        'N21-186',
        'N21-198',
        'N21-202',
        'N21-205',
        'N21-220',
        'N21-229',
        'N21-247',
        'N21-259',
        'N21-287',
        'N21-317',
        'N21-324',
        'N21-327',
        'N22-005',
        'N22-013',
        'N22-035',
        'N22-114',
        'N22-123',
        'N22-125',
        'N22-135',
        'N22-164',
        'N22-190',
        'N22-225',
        'N22-227',
        'N22-272',
        'N22-285',
    ],
    'M': [
        'N20-002',
        'N20-077',
        'N20-080',
        'N20-089',
        'N20-105',
        'N20-108',
        'N20-133',
        'N20-144',
        'N20-180',
        'N20-183',
        'N20-205',
        'N20-211',
        'N20-212',
        'N20-216',
        'N20-219',
        'N20-222',
        'N20-247',
        'N20-290',
        'N21-004',
        'N21-012',
        'N21-066',
        'N21-084',
        'N21-100',
        'N21-129',
        'N21-189',
        'N21-254',
        'N21-267',
        'N21-280',
        'N21-291',
        'N21-295',
        'N21-332',
        'N22-001',
        'N22-018',
        'N22-036',
        'N22-085',
        'N22-115',
        'N22-117',
        'N22-140',
        'N22-216',
        'N22-230',
        'N22-257',
        'N22-276',
        'N22-294',
    ]
}
base_src_dir = '/var/run/media/ken/ENDA2/Nバックアップ/'
base_dest_dir = '/var/run/media/ken/8t/Public/Datasets/chouwa/'

J = os.path.join

class CMD(BaseCLI):
    class CopyArgs(BaseCLI.CommonArgs):
        target: str = Field(..., cli=('--target', '-t'), regex=r'^[L|M|G|A|O]$')
        dryrun: bool = Field(False, cli=('--dryrun', '-s'))

    def run_copy(self, a):
        tq = tqdm(targets[a.target])
        for t in tq:
            N = t[0:3]
            src_dir = J(base_src_dir, N, t)
            dest_dir = J(base_dest_dir, a.target, 'todo', t)
            os.makedirs(dest_dir, exist_ok=True)
            for src_path in sorted(glob(J(src_dir, '*.ndpi'))):
                if not re.match(r'.*HE.*', src_path):
                    continue
                name = os.path.basename(src_path)
                dest_path = J(dest_dir, name)
                tq.set_description(f'copying {name}')
                if os.path.exists(dest_path) and filecmp.cmp(src_path, dest_path):
                    continue
                if a.dryrun:
                    time.sleep(1)
                else:
                    shutil.copy2(src_path, dest_dir)
            tq.refresh()


    def run_p(self, a):
        t = tqdm(range(100))
        for i in t:
            for x in range(10):
                t.set_description(f'{i}: {x}')
                time.sleep(0.1)



cmd = CMD()
cmd.run()
