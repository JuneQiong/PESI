import os

# lr = 1
# rat_lr = [0.01, 0.001, 0.0005]
# gamma = [0.15, 0.1, 0.2, 0.25]
# print('test lr con lr on amo dataset')
#
# count = 0


# for i in r nd---main_newmuti_xrloss.py  --num_ {}'.format(i))

# for i in range(1):
#     data_path = '../data/yelp/reviews.pickle'
#     index_dir = '../data/yelp/1/'
#     checkpoint = 'yelp_test_lamda5_{}'.format(i)
#     outf = 'yelp_test_lamda5_{}'.format(i)
#
#     print('-----begin---main_distangle_test.py  --num_ {}'.format(i))
#     # os.system(
#     #     'python main_init.py --checkpoint={} --outf={}  --lr={} --con_lr={} --rating_reg={} --con_reg={} --conself_reg={}'.format(checkpoint, outf, l, cl, rr, cr, cr))
#     os.system(
#         'python main_distangle_newmuti.py --checkpoint %s --outf %s --data_path %s --index_dir  %s'%(
#             checkpoint, outf, data_path, index_dir))
#     print('-----end---main_distangle_test.py  --num_ {}'.format(i))
#


# ganma = [0.1, 0.15, 0.2, 0.25]
# for i in [1,2]:
#     checkpoint = 'mov_test_r101_inwei01_gam015_{}'.format(i)
#     outf = 'mov_test_r101_inwei01_gam015_{}'.format(i)
#
#     print('-----begin---main_distangle_test.py  --num_ {}'.format(i))
#     # os.system(
#     #     'python main_init.py --checkpoint={} --outf={}  --lr={} --con_lr={} --rating_reg={} --con_reg={} --conself_reg={}'.format(checkpoint, outf, l, cl, rr, cr, cr))
#     os.system(
#         'python main_distangle_newmuti.py --checkpoint %s --outf %s '%(
#             checkpoint, outf))
#     print('-----end---main_distangle_test.py  --num_ {}'.format(i))


# for i in [5, 6]:
#     data_path = '../data/TripAdvisor/reviews.pickle'
#     index_dir = '../data/TripAdvisor/1/'
#     checkpoint = 'trip_test_losswight105_wei01_lamda{}'.format(i)
#     outf = 'trip_test_losswight105_wei01_lamda{}'.format(i)
#
#     print('-----begin---main_distangle_test.py  --num_ {}'.format(i))
#     # os.system(
#     #     'python main_init.py --checkpoint={} --outf={}  --lr={} --con_lr={} --rating_reg={} --con_reg={} --conself_reg={}'.format(checkpoint, outf, l, cl, rr, cr, cr))
#     os.system(
#         'python main_distangle_newmuti.py --checkpoint %s --outf %s --data_path %s --index_dir %s --lamda %s '%(
#             checkpoint, outf, data_path, index_dir, i))
#     print('-----end---main_distangle_test.py  --num_ {}'.format(i))

for i in [16, 32, 64, 128, 256, 1024]:
    # data_path = '../data/TripAdvisor/reviews.pickle'
    # index_dir = '../data/TripAdvisor/1/'
    checkpoint = 'mov_embed_{}'.format(i)
    outf = 'mov_embed_{}'.format(i)

    print('-----begin---main_distangle_test.py  --num_ {}'.format(i))
    # os.system(
    #     'python main_init.py --checkpoint={} --outf={}  --lr={} --con_lr={} --rating_reg={} --con_reg={} --conself_reg={}'.format(checkpoint, outf, l, cl, rr, cr, cr))
    os.system(
        'python main_distangle_newmuti.py --checkpoint %s --outf %s --emsize %s' % (
            checkpoint, outf, i))
    print('-----end---main_distangle_test.py  --num_ {}'.format(i))

for i in [2048]:
    data_path = '../data/TripAdvisor/reviews.pickle'
    index_dir = '../data/TripAdvisor/1/'
    checkpoint = 'trip_words_{}'.format(i)
    outf = 'trip_words_{}'.format(i)

    print('-----begin---main_distangle_test.py  --num_ {}'.format(i))
    # os.system(
    #     'python main_init.py --checkpoint={} --outf={}  --lr={} --con_lr={} --rating_reg={} --con_reg={} --conself_reg={}'.format(checkpoint, outf, l, cl, rr, cr, cr))
    os.system(
        'python main_distangle_newmuti.py --checkpoint %s --outf %s --emsize %s' % (
            checkpoint, outf, i))
    print('-----end---main_distangle_test.py  --num_ {}'.format(i))

# ganma = [0.15, 0.1, 0.2]
# for i in [0]:
#     data_path = '../data/yelp/reviews.pickle'
#     index_dir = '../data/yelp/1/'
#     checkpoint = 'yelp_test_r105_inwei01_gam02_{}'.format(i)
#     outf = 'yelp_test_r105_inwei01_gam02_{}'.format(i)
#
#     print('-----begin---main_distangle_test.py  --num_ {}'.format(i))
#     # os.system(
#     #     'python main_init.py --checkpoint={} --outf={}  --lr={} --con_lr={} --rating_reg={} --con_reg={} --conself_reg={}'.format(checkpoint, outf, l, cl, rr, cr, cr))
#     os.system(
#         'python main_distangle_newmuti.py --checkpoint %s --outf %s --data_path %s --index_dir %s '%(
#             checkpoint, outf, data_path, index_dir))
#     print('-----end---main_distangle_test.py  --num_ {}'.format(i))

# for i in [0, 1, 2]:
#     # data_path = '../data/Amazon/Clothing Shoes and Jewelry/reviews.pickle'
#     # index_dir = '../data/Amazon/Clothing Shoes and Jewelry/1/'
#     checkpoint = 'clo_r101_inwei01_gam015_lamda4_{}'.format(i)
#     outf = 'clo_r101_inwei01_gam015_lamda4_{}'.format(i)
#
#     print('-----begin---main_distangle_test.py  --num_ {}'.format(i))
#     # os.system(
#     #     'python main_init.py --checkpoint={} --outf={}  --lr={} --con_lr={} --rating_reg={} --con_reg={} --conself_reg={}'.format(checkpoint, outf, l, cl, rr, cr, cr))
#     os.system(
#         'python main_distangle_newmuti.py --checkpoint %s --outf %s '%(
#             checkpoint, outf))
#     print('-----end---main_distangle_test.py  --num_ {}'.format(i))