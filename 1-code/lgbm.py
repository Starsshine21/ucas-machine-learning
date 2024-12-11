import warnings
warnings.simplefilter('ignore')

import os
import gc

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error

import lightgbm as lgb

# 玩家角色表

roles = pd.read_csv('./data/role_id.csv')

# 共七天, roles 表填充完整
dfs = []
for i in range(2, 9):
    tmp = roles.copy()
    tmp['day'] = i
    dfs.append(tmp)
data = pd.concat(dfs).reset_index(drop=True)

# 货币消耗表
consume = pd.read_csv('./data/role_consume_op.csv')
consume['dt'] = pd.to_datetime(consume['dt'])
consume['day'] = consume['dt'].dt.day

# 货币消耗按天合并
# TODO: mtime 可以做时差衍生特征, 其他表也是一样的
for i in range(1, 5):
    for m in ['count', 'sum']:
        tmp = consume.groupby(['role_id', 'day'])[f'use_t{i}'].agg(m).to_frame(name=f'use_t{i}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')


# 升级表

evolve = pd.read_csv('./data/role_evolve_op.csv')
evolve['dt'] = pd.to_datetime(evolve['dt'])
evolve['day'] = evolve['dt'].dt.day
evolve['n_level_up'] = evolve['new_lv'] - evolve['old_lv']
evolve = evolve.rename(columns={'num': 'lv_consume_item_num'})

for col in ['type', 'item_id']:
    for m in ['count', 'nunique']:
        tmp = evolve.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')
for col in ['lv_consume_item_num', 'n_level_up']:
    for m in ['sum', 'mean']:
        tmp = evolve.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')
        

# 副本表
# TODO: 这个表信息比较多, 可以多挖掘

fb = pd.read_csv('./data/role_fb_op.csv')
fb['dt'] = pd.to_datetime(fb['dt'])
fb['day'] = fb['dt'].dt.day
fb['fb_used_time'] = fb['finish_time'] - fb['start_time']

for col in ['fb_id', 'fb_type']:
    for m in ['count', 'nunique']:
        tmp = fb.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')
for col in ['fb_used_time', 'exp']:
    for m in ['sum', 'mean']:
        tmp = fb.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')
        
tmp = fb.groupby(['role_id', 'day'])['fb_result'].value_counts().reset_index(name='fb_result_count')        
for i in [0, 1, 2]:
    tt = tmp[tmp['fb_result'] == i]
    tt.columns = list(tt.columns[:-1]) + ['fb_result%d_count'%i]
    data = data.merge(tt[['role_id', 'day', 'fb_result%d_count'%i]], on=['role_id', 'day'], how='left')
# 任务系统表

mission = pd.read_csv('./data/role_mission_op.csv')
mission['dt'] = pd.to_datetime(mission['dt'])
mission['day'] = mission['dt'].dt.day

for col in ['mission_id', 'mission_type']:
    for m in ['count', 'nunique']:
        tmp = mission.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')
        

# 玩家离线表
# TODO: 可以做很多时间、坐标方面的特征

offline = pd.read_csv('./data/role_offline_op.csv')
offline['dt'] = pd.to_datetime(mission['dt'])
offline['day'] = offline['dt'].dt.day
offline['online_durations'] = offline['offline'] - offline['online']

for col in ['reason', 'map_id']:
    for m in ['count', 'nunique']:
        tmp = offline.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')
        
for col in ['online_durations']:
    for m in ['mean', 'sum']:
        tmp = offline.groupby(['role_id', 'day'])[col].agg(m).to_frame(name=f'{col}_day_{m}').reset_index()
        data = data.merge(tmp, on=['role_id', 'day'], how='left')

# 付费表
pay = pd.read_csv('./data/role_pay.csv')
pay['dt'] = pd.to_datetime(pay['dt'])
pay['day'] = pay['dt'].dt.day
tmp = pay.groupby(['role_id', 'day'])['pay'].agg('sum').to_frame(name='pay_sum_day').reset_index()
data = data.merge(tmp, on=['role_id', 'day'], how='left')
data['pay_sum_day'].fillna(0., inplace=True)
data['pay_log_fea'] = np.log1p(data['pay_sum_day'])


# 训练集 day 2,3,4,5 -> 标签 day 6 pay_sum
xy_train = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(2, 6)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    xy_train = xy_train.merge(tmp, on='role_id')

# 验证集 day 3,4,5,6 -> 标签 day 7 pay_sum
xy_valid = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(3, 7)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    xy_valid = xy_valid.merge(tmp, on='role_id')
    
# 测试集 day 4,5,6,7  -> 标签 day 8 pay_sum
xy_test = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(4, 8)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    xy_test = xy_test.merge(tmp, on='role_id')


# 训练集 day == 7 pay_sum
# 验证集 day == 8 pay_sum
tmp = data[data.day == 6].copy().reset_index(drop=True)
tmp = tmp[['role_id', 'pay_sum_day']]
tmp = tmp.rename(columns={'pay_sum_day' : 'pay'})
xy_train = xy_train.merge(tmp, on='role_id', how='left')
xy_train['pay'].fillna(0., inplace=True)
xy_train['pay_log'] = np.log1p(xy_train['pay'])


tmp = data[data.day == 7].copy().reset_index(drop=True)
tmp = tmp[['role_id', 'pay_sum_day']]
tmp = tmp.rename(columns={'pay_sum_day' : 'pay'})
xy_valid = xy_valid.merge(tmp, on='role_id', how='left')
xy_valid['pay'].fillna(0., inplace=True)
xy_valid['pay_log'] = np.log1p(xy_valid['pay'])

params = {
    'objective': 'regression',
    'metric': {'rmse'},
    'boosting_type' : 'gbdt',
    'learning_rate': 0.05,
    'max_depth' : 5,
    'num_leaves' : 8,
    'feature_fraction' : 1,
    'subsample' : 0.8,
    'seed' : 114,
    'num_iterations' : 3000,
    'nthread' : -1,
    'verbose' : -1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.2
}
features = [col for col in xy_train.columns if col not in ['role_id', 'pay', 'pay_log']]
#features = [col for col in xy_train.columns if col not in ['role_id', 'pay', 'pay_log', 'pay_sum_day_day0','pay_sum_day_day1','pay_sum_day_day2','pay_sum_day_day3']]
#features = [col for col in xy_train.columns if col not in ['role_id', 'pay', 'pay_log', 'pay_log_fea_day0','pay_log_fea_day1','pay_log_fea_day2','pay_log_fea_day3']]

# features = ['pay_sum_day_day3', 'use_t2_day_sum_day3','use_t2_day_sum_day2', 'item_id_day_nunique_day3','fb_used_time_day_sum_day3','pay_sum_day_day2',
#             'n_level_up_day_sum_day3', 'use_t1_day_count_day2', 'n_level_up_day_mean_day3','lv_consume_item_num_day_sum_day3','use_t2_day_sum_day1','use_t1_day_sum_day1']

def train(xy_train, xy_valid, label, params, features):
    train_label = xy_train[label].values
    train_feat = xy_train[features]

    valid_label = xy_valid[label].values
    valid_feat = xy_valid[features]
    gc.collect()

    trn_data = lgb.Dataset(train_feat, label=train_label)
    val_data = lgb.Dataset(valid_feat, label=valid_label)
    clf = lgb.train(params,
                    trn_data,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=50,
                    early_stopping_rounds=100)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    fold_importance_df = fold_importance_df.sort_values(by='importance', ascending=False)
    print(fold_importance_df[:30])

    xy_valid['{}_preds'.format(label)] = clf.predict(valid_feat, num_iteration=clf.best_iteration)
    xy_valid['{}_preds'.format(label)] = xy_valid['{}_preds'.format(label)].clip(lower=0.)
    
    result = mean_squared_log_error(np.expm1(xy_valid[label]), 
                                    np.expm1(xy_valid['{}_preds'.format(label)]))
    
    return clf, result

clf_valid, result_valid = train(xy_train, xy_valid, 'pay_log', params, features)

# 用 4,5,6,7,8 重新训练模型
params['num_iterations'] = clf_valid.best_iteration
clf_test, _ = train(xy_valid, xy_valid, 'pay_log', params, features)

xy_test['pay'] = np.expm1(clf_test.predict(xy_test[features]))
xy_test['pay'] = xy_test['pay'].clip(lower=0.)

y_test_pred = pd.read_csv('./data/submission_sample.csv')
tmp = xy_test[['role_id', 'pay']].copy()
y_test_pred = y_test_pred[['role_id']].merge(tmp, on='role_id', how='left')

y_test = data[data.day == 8].copy().reset_index(drop=True)
y_test_real = pd.read_csv('./data/submission_sample.csv')
y_test_real = y_test_real[['role_id']].merge(y_test, on = 'role_id', how='left')


#y_test_pred['pay'] = np.where(y_test_pred['pay'] < 0.1, 0, y_test_pred['pay'])
y_test_pred['pay'] = np.where((y_test_pred['pay'] > 3) & (y_test_pred['pay'] < 15), 6, y_test_pred['pay'])
y_test_pred['pay'] = np.where((y_test_pred['pay'] > 25) & (y_test_pred['pay'] < 30), 30, y_test_pred['pay'])


msle = mean_squared_log_error(y_test_real['pay_sum_day'], y_test_pred['pay'])
msle = 1 / (msle + 1)
print(f'score(pay_log): {msle}')
