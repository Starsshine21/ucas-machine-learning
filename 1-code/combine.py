from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

## 已经成功实现的线性模型 pay进行了log处理 正确率较高

import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd


from sklearn.metrics import mean_squared_log_error
import numpy as np


from sklearn.metrics import mean_squared_log_error



# 玩家角色表
roles = pd.read_csv('./data/role_id.csv')
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

# 货币消耗按天合并 这里统计出每个用户每天每个货币的消耗量和消耗次数
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
#统计升级类型的类型数和次数  消耗物品的种类、次数和数量  每次升级花费的材料数 每次操作的升级数
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
#pay_sum_to_day



# 现在我们可以把问题转为用前4天的行为来预测第5天的付费
# 训练集 day 2,3,4,5 -> 标签 day 6 pay_sum
X_train = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(2, 6)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    X_train = X_train.merge(tmp, on='role_id')
    
# 验证集 day 3,4,5,6
X_valid = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(3, 7)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    X_valid = X_valid.merge(tmp, on='role_id')

# 测试集 day 4,5,6,7
X_test = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(4, 8)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    X_test = X_test.merge(tmp, on='role_id')

y_train = data[data.day == 6].copy().reset_index(drop=True)
y_train = y_train[['role_id','pay_log_fea']]
y_train = y_train.rename(columns={'pay_log_fea':'pay_log'})
y_train = y_train['pay_log']
y_train.fillna(0, inplace=True)

y_valid = data[data.day == 7].copy().reset_index(drop=True)
y_valid = y_valid[['role_id','pay_log_fea']]
y_valid = y_valid.rename(columns={'pay_log_fea':'pay_log'})
y_valid = y_valid['pay_log']
y_valid.fillna(0, inplace=True)


# features = [col for col in X_train.columns if col not in ['role_id', 'day', 'create_time', 'pay_log']] #前一天的pay_log和pay_sum_day也作为特征
features = ['pay_sum_day_day3', 'use_t2_day_sum_day3','use_t2_day_sum_day2', 'item_id_day_nunique_day3','fb_used_time_day_sum_day3','pay_sum_day_day2',
             'n_level_up_day_sum_day3', 'use_t1_day_count_day2', 'n_level_up_day_mean_day3','lv_consume_item_num_day_sum_day3','use_t2_day_sum_day1','use_t1_day_sum_day1']

X_train = X_train[features]
X_train.fillna(0, inplace=True)

X_valid = X_valid[features]
X_valid.fillna(0, inplace=True)

X_test = X_test[features]
X_test.fillna(0, inplace=True)

num_features = len(features)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, build_fn, **kwargs):
        self.build_fn = build_fn
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y, **fit_params):
        self.model = self.build_fn(**self.kwargs)
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.model.predict(X)


def create_neural_network():
    model = Sequential()
    model.add(Dense(32, input_dim=num_features, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

num_features = len(features)


class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, build_fn, **kwargs):
        self.build_fn = build_fn
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y, **fit_params):
        self.model = self.build_fn(**self.kwargs)
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.model.predict(X)

def create_neural_network(epochs=100, batch_size=32, verbose=0):
    model = Sequential()
    model.add(Dense(32, input_dim=num_features, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

base_models = [
    ('linear', LinearRegression()),
    ('neural_network', KerasRegressorWrapper(build_fn=create_neural_network, epochs=100, batch_size=32, verbose=0))
]

meta_model = LinearRegression()  
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)


for name, model in base_models:
    if name == 'neural_network':
        model.fit(X_train[features].to_numpy(), y_train, epochs=100, batch_size=32, verbose=0)
    else:
        model.fit(X_train[features], y_train)


for name, model in base_models:
    if name == 'neural_network':
        y_pred = model.predict(X_valid[features].to_numpy())
    else:
        y_pred = model.predict(X_valid[features])
    y_valid = np.clip(y_valid, 0, None)
    y_pred = np.clip(y_pred, 0, None)

    msle = mean_squared_log_error(y_valid, y_pred)
    msle = 1 / (msle + 1)
    print(f'score({name}): {msle}')


stacking_model.fit(X_valid, y_valid)


y_test_pred = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
y_test_pred['pay_log'] = stacking_model.predict(X_test)

y_test_pred['pay'] = np.expm1(y_test_pred['pay_log'])
y_test_pred['pay'] = y_test_pred['pay'].clip(lower=0.)

tmp = pd.read_csv('./data/submission_sample.csv')
y_test_pred = tmp[['role_id']].merge(y_test_pred, on='role_id', how='left')
# y_test_pred['pay'] = np.where((sub['pay'] > 3) & (y_test_pred['pay'] < 15), 6, y_test_pred['pay'])
# y_test_pred['pay'] = np.where((sub['pay'] > 25) & (y_test_pred['pay'] < 30), 30, y_test_pred['pay'])
y_test_pred = y_test_pred['pay']
y_test_pred.fillna(0, inplace=True)

y_test_true = data[data.day == 8].copy().reset_index(drop=True)
y_test_true = y_test_true[['role_id', 'pay_sum_day']]
y_test_true = y_test_true.rename(columns={'pay_sum_day':'pay'})
y_test_true.fillna(0, inplace=True)

tmp = pd.read_csv('./data/submission_sample.csv')
y_test_true = tmp[['role_id']].merge(y_test_true, on='role_id', how='left')
y_test_true = y_test_true['pay']



msle = mean_squared_log_error(y_test_true, y_test_pred)
msle = 1 / (msle + 1)
print(f'score(pay_log): {msle}')