import os
import pdb

seed = [0, 11, 111, 1111]


print('test lr con lr on amo dataset')

count = 0

for se in seed:

                checkpoint = 'tuneparam_amoclo_seed_{}'.format(se)
                outf ='tuneparam_amoclo_seed_{}'.format(se)

                print('-----begin---main_init.py --seed{}'.format(se))
                # os.system(
                #     'python main_init.py --checkpoint={} --outf={}  --lr={} --con_lr={} --rating_reg={} --con_reg={} --conself_reg={}'.format(checkpoint, outf, l, cl, rr, cr, cr))
                os.system('python main_init.py --seed %s --checkpoint %s --outf %s'%(se, checkpoint, outf))
                print('-----end---main_init.py  --seed_{}'.format(se))
