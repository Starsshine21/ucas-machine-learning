## 已经成功实现的线性模型 pay进行了log处理 正确率较高

import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_log_error
import numpy as np



class Linear_Model:
    def __init__(self, learning_rate=0.1, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    def initialize_parameters(self, num_features):
        self.weights = np.zeros(num_features)
        self.bias = 0.0
    def forward(self, X):
        return np.dot(X, self.weights) + self.bias
    def compute_loss(self, X, y):
        m = len(y)
        predictions = self.forward(X)
        loss = np.sum((predictions - y) ** 2) / (2 * m)
        return loss
    def backward(self, X, y, predictions):
        m = len(y)
        dw = np.dot(X.T, (predictions - y)) / m
        db = np.sum(predictions - y) / m
        return dw, db
    def update_parameters(self, dw, db):
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.initialize_parameters(num_features)
        for i in range(self.num_iterations):
            predictions = self.forward(X)
            loss = self.compute_loss(X, y)
            dw, db = self.backward(X, y, predictions)
            self.update_parameters(dw, db)
    def predict(self, X):
        return self.forward(X)


from sklearn.metrics import mean_squared_log_error


class Linear_Model_OLS:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        self.weights = np.linalg.pinv(X_b).dot(y)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha 
        self.weights = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # 使用岭回归的最小二乘法进行参数估计
        identity_matrix = np.eye(X_b.shape[1])
        self.weights = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * identity_matrix).dot(X_b.T).dot(y)

    def predict(self, X):
        # 添加截距项
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha  # 正则化参数
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.weights = None

    def soft_threshold(self, x, alpha):
        return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m, n = X_b.shape
        self.weights = np.zeros(n)
        old_weights = self.weights.copy()

        for _ in range(self.max_iter):
            for j in range(n):
                X_j = X_b[:, j]
                X_j_residual = X_b[:, np.arange(n) != j].dot(self.weights[np.arange(n) != j])
                rho = X_j.T.dot(y - X_j_residual)
                z = X_j.T.dot(X_j)
                self.weights[j] = self.soft_threshold(rho, self.alpha) / z if z != 0 else 0
            if np.sum(np.abs(self.weights - old_weights)) < self.tol:
                break
            old_weights = self.weights.copy()

    def predict(self, X):

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)


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
# 训练集 day 2,3,4,5,6 -> 标签 day 6 pay_sum
X_train = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(2, 7)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    X_train = X_train.merge(tmp, on='role_id')
    
# 测试集 day 3,4,5,6,7
X_test = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
for i, d in enumerate(range(3, 8)):
    tmp = data[data.day == d].copy().reset_index(drop=True)
    tmp.drop(['create_time', 'day'], axis=1, inplace=True)
    tmp.columns = ['role_id'] + [f'{c}_day{i}' for c in tmp.columns[1:]]
    X_test = X_test.merge(tmp, on='role_id')

y_train = data[data.day == 7].copy().reset_index(drop=True)
y_train = y_train[['role_id','pay_log_fea']]
y_train = y_train.rename(columns={'pay_log_fea':'pay_log'})
y_train = y_train['pay_log']
y_train.fillna(0, inplace=True)



features = [col for col in X_train.columns if col not in ['role_id', 'day', 'create_time', 'pay_log']] #前一天的pay_log和pay_sum_day也作为特征
# features = ['pay_sum_day_day3', 'use_t2_day_sum_day3','use_t2_day_sum_day2', 'item_id_day_nunique_day3','fb_used_time_day_sum_day3','pay_sum_day_day2',
#             'n_level_up_day_sum_day3', 'use_t1_day_count_day2', 'n_level_up_day_mean_day3','lv_consume_item_num_day_sum_day3','use_t2_day_sum_day1','use_t1_day_sum_day1']

X_train = X_train[features]
X_train.fillna(0, inplace=True)

X_test = X_test[features]
X_test.fillna(0, inplace=True)

from sklearn.linear_model import LinearRegression

# 创建线性回归模型并训练
model = LinearRegression()
# model = Linear_Model()
# model = Linear_Model_OLS()
# model = RidgeRegression()
# model = LassoRegression()
model.fit(X_train, y_train)

y_test_pred = pd.DataFrame({'role_id': data.role_id.unique().tolist()})
y_test_pred['pay_log'] = model.predict(X_test)

y_test_pred['pay'] = np.expm1(y_test_pred['pay_log'])
y_test_pred['pay'] = y_test_pred['pay'].clip(lower=0.)

tmp = pd.read_csv('./data/submission_sample.csv')
y_test_pred = tmp[['role_id']].merge(y_test_pred, on='role_id', how='left')
# y_test_pred['pay'] = np.where(tmp['pay'] < 0.1, 0, y_test_pred['pay'])
# y_test_pred['pay'] = np.where((tmp['pay'] > 3) & (y_test_pred['pay'] < 15), 6, y_test_pred['pay'])
# y_test_pred['pay'] = np.where((tmp['pay'] > 25) & (y_test_pred['pay'] < 30), 30, y_test_pred['pay'])




y_test_true = data[data.day == 8].copy().reset_index(drop=True)
y_test_true = y_test_true[['role_id', 'pay_sum_day']]
y_test_true = y_test_true.rename(columns={'pay_sum_day':'pay'})
y_test_true.fillna(0, inplace=True)

# y_test_pred['pay'] = np.where(y_test_pred['pay'] < 0.1, 0, y_test_pred['pay'])
y_test_pred['pay'] = np.where((y_test_pred['pay'] > 3) & (y_test_pred['pay'] < 15), 6, y_test_pred['pay'])
y_test_pred['pay'] = np.where((y_test_pred['pay'] > 25) & (y_test_pred['pay'] < 30), 30, y_test_pred['pay'])

y_test_pred = y_test_pred['pay']
y_test_pred.fillna(0, inplace=True)

tmp = pd.read_csv('./data/submission_sample.csv')
y_test_true = tmp[['role_id']].merge(y_test_true, on='role_id', how='left')
y_test_true = y_test_true['pay']
# print(len(y_test_pred), len(y_test_true))

msle = mean_squared_log_error(y_test_true, y_test_pred)
msle = 1 / (msle + 1)
print(f'score(pay_log): {msle}')

