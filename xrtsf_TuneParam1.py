import os

# lr = 1
# rat_lr = [0.01, 0.001, 0.0005]
# gamma = [0.15, 0.1, 0.2, 0.25]
# print('test lr con lr on amo dataset')
#
# count = 0

# for rl in rat_lr:
#     for ga in gamma:
#             checkpoint = 'tuneparam_amamov_lr_{}rat_lr{}_gamma{}'.format(lr, rl, ga)
#             outf = 'tuneparam_amamov_lr_{}rat_lr{}_gamma{}'.format(lr, rl, ga)
#
#             print('-----begin---main_init.py  --lr {} --rat_lr {} --gamma{}'
#                 '--checkpoint {} --outf {} " '.format(lr, rl, ga, checkpoint, outf))
#             # os.system(
#             #     'python main_init.py --checkpoint={} --outf={}  --lr={} --con_lr={} --rating_reg={} --con_reg={} --conself_reg={}'.format(checkpoint, outf, l, cl, rr, cr, cr))
#             os.system(
#                 'python main_distangle_2tsf.py --checkpoint %s --outf %s  --lr %s --rat_lr %s --gamma %s'%(
#                     checkpoint, outf, lr, rl, ga))
#             print('-----end---main_init.py  --lr {} --rat_lr {} --gamma{}'
#                   '--checkpoint {} --outf {} " '.format(lr, rl, ga, checkpoint, outf))

# for i in r nd---main_newmuti_xrloss.py  --num_ {}'.format(i))
#
for i in range(3):

    checkpoint = 'mov_test_{}'.format(i)
    outf = 'mov_test_{}'.format(i)

    print('-----begin---main_distangle_test.py  --num_ {}'.format(i))
    # os.system(
    #     'python main_init.py --checkpoint={} --outf={}  --lr={} --con_lr={} --rating_reg={} --con_reg={} --conself_reg={}'.format(checkpoint, outf, l, cl, rr, cr, cr))
    os.system(
        'python main_distangle_xr_tsf.py --checkpoint %s --outf %s '%(
            checkpoint, outf))
    print('-----end---main_distangle_test.py  --num_ {}'.format(i))

# for i in range(3):
#     data_path = '../data/TripAdvisor/reviews.pickle'
#     index_dir = '../data/TripAdvisor/1/'
#     checkpoint = 'trip_test_{}'.format(i)
#     outf = 'trip_test_{}'.format(i)
#
#     print('-----begin---main_distangle_test.py  --num_ {}'.format(i))
#     # os.system(
#     #     'python main_init.py --checkpoint={} --outf={}  --lr={} --con_lr={} --rating_reg={} --con_reg={} --conself_reg={}'.format(checkpoint, outf, l, cl, rr, cr, cr))
#     os.system(
#         'python main_distangle_xr_tsf.py --checkpoint %s --outf %s --data_path %s --index_dir  %s'%(
#             checkpoint, outf, data_path, index_dir))
#     print('-----end---main_distangle_test.py  --num_ {}'.format(i))

# for i in range(3):
#     data_path = '../data/yelp/reviews.pickle'
#     index_dir = '../data/yelp/1/'
#     checkpoint = 'yelp_test_{}'.format(i)
#     outf = 'yelp_test_{}'.format(i)
#
#     print('-----begin---main_distangle_test.py  --num_ {}'.format(i))
#     # os.system(
#     #     'python main_init.py --checkpoint={} --outf={}  --lr={} --con_lr={} --rating_reg={} --con_reg={} --conself_reg={}'.format(checkpoint, outf, l, cl, rr, cr, cr))
#     os.system(
#         'python main_distangle_xr_tsf.py --checkpoint %s --outf %s --data_path %s --index_dir  %s'%(
#             checkpoint, outf, data_path, index_dir))
#     print('-----end---main_distangle_test.py  --num_ {}'.format(i))