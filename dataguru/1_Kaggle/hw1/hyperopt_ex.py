
import hyperopt
from hyperopt import tpe, fmin, hp
# Minimize the material used in a typical coca-cola can
best = fmin(
fn=lambda x: 6.28 * x**2 + 500/x, space=hp.uniform('x', 0, 10),algo=tpe.suggest, max_evals=100)

print('the best radius of a typical can of 500ml is: ', best,' cm')
