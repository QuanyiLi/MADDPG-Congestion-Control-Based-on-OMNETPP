import collections
import numpy as np
import os
import tensorflow as tf


def sum(x, axis=None, keepdims=False):  # 降维求和
    return tf.reduce_sum(x, axis=None if axis is None else [axis], keep_dims=keepdims)


def mean(x, axis=None, keepdims=False):  # 降维平均
    return tf.reduce_mean(x, axis=None if axis is None else [axis], keep_dims=keepdims)


def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)  # 均值
    return mean(tf.square(x - meanx), axis=axis, keepdims=keepdims)  # 返回方差


def std(x, axis=None, keepdims=False):  # 标准差
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))


def max(x, axis=None, keepdims=False):  # 降维最大值
    return tf.reduce_max(x, axis=None if axis is None else [axis], keep_dims=keepdims)


def min(x, axis=None, keepdims=False):  # 降维最小值
    return tf.reduce_min(x, axis=None if axis is None else [axis], keep_dims=keepdims)


def concatenate(arrs, axis=0):  # 矩阵拼接
    return tf.concat(axis=axis, values=arrs)


def argmax(x, axis=None):  # 返回最大值的索引号index
    return tf.argmax(x, axis=axis)


def softmax(x, axis=None):  # softmax函数
    return tf.nn.softmax(x, axis=axis)


# ================================================================
# Misc
# ================================================================


def is_placeholder(x):
    return type(x) is tf.Tensor and len(x.op.inputs) == 0  # x是否是张量 ∪ 输入的长度等于0时整个式子为一
    # 即x这个张量需要几个输入例如:a=tf.tensor(1,nmae='a') b=tf.tensor(2,name='b'),那么len(a.op.inputs)=0
    # c=a*b 则张量c的len（c.op.inputs）=2!!!!!!!!!!!!!,该式子返回bool值表示这是否是一个不需要别的张量输入的张量


# ================================================================
# Inputs
# ================================================================


class TfInput(object):  # 作为父类，用于下面子类函数，并且重构方法，
    # 父类方法未定义用raise notimplemented来占位，使得程序可以寻找其他方法完成
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplemented()

    def make_feed_dict(data):
        """Given data input it to the placeholder(s)."""
        raise NotImplemented()


class PlacholderTfInput(TfInput):  # 上面类的子类，下面类的父类
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholder."""
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):  # 重构
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}  # 用于填充占位符字典


class BatchInput(PlacholderTfInput):  # 一批张量的占位符
    def __init__(self, shape, dtype=tf.float32, name=None):
        """Creates a placeholder for a batch of tensors of a given shape and dtype

        Parameters
        ----------
        shape: [int]
            shape of a single elemenet of the batch(每一个张量的shape）
        dtype: tf.dtype
            number representation used for tensor contents
        name: str
            name of the underlying placeholder
        """
        super().__init__(tf.placeholder(dtype, [None] + list(shape), name=name))  # 父类继承
        # 创建一批形状为shape的张量，数据类型为dtype具有self.name属性与self.placeholder属性（等于输入的这一批张量）
        # 并且继承了父类的get方法获取该张量，并且由字典填充方法为该占位符填充数据，用get方法获得这批张量


class Uint8Input(PlacholderTfInput):  # 继承于填充张量，转为32位浮点数，并且除以255将float数据转到0~1之间，创造一批uint8格式张量
    def __init__(self, shape, name=None):
        """Takes input in uint8 format which is cast to float32 and divided by 255（uint8为0~255之间的数）
        before passing it to the model.

        On GPU this ensures lower data transfer times.

        Parameters
        ----------
        shape: [int]
            shape of the tensor.
        name: str
            name of the underlying placeholder
        """

        super().__init__(tf.placeholder(tf.uint8, [None] + list(shape), name=name))
        self._shape = shape  # shape属性
        self._output = tf.cast(super().get(), tf.float32) / 255.0  # 类型转换输出uint8型的一批张量

    def get(self):
        return self._output  # 可用_output属性获得这批张量的uint8形式或用get方法获得，get方法重构


def ensure_tf_input(thing):
    """Takes either tf.placeholder of TfInput and outputs equivalent TfInput"""
    if isinstance(thing, TfInput):  # 输入已经是自订的tfinput类直接返回即可，该方法可将所有张量转为自定的张量类
        return thing
    elif is_placeholder(thing):  # 如果是不需要输入的张量，转为一个自定的placeholderinput类 注：placeholder占位符也是一个张量tf.Tensor
        return PlacholderTfInput(thing)
    else:
        raise ValueError("Must be a placeholder or TfInput")  # 否则报错


# ================================================================
# Mathematical utils
# ================================================================


def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


# ================================================================
# Optimizer utils
# ================================================================


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    # http://www.mamicode.com/info-detail-2375709.html
    if clip_val is None:
        return optimizer.minimize(objective, var_list=var_list)
    else:  # 完成的功能等于上式
        gradients = optimizer.compute_gradients(objective, var_list=var_list)  # 计算var_list中的变量的梯度，结果是梯度，变量构成的元组
        for i, (grad, var) in enumerate(gradients):  # 返回gradients中每个变量的梯度的标及其
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)  # 梯度存在的话进行梯度裁剪，防止梯度爆炸，小于10的不会被裁剪，大于10
                # 会乘以10并除以l2范数
        return optimizer.apply_gradients(gradients)  # 应用梯度下降，整个函数依据是否有clip约束用两种不同的最小化方法


# ================================================================
# Global session
# ================================================================

def get_session():
    """Returns recently made Tensorflow session"""
    return tf.get_default_session()  # 获得当前默认会话


def make_session(num_cpu):
    """Returns a session that will use <num_cpu> CPU's only"""
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,  # 一个操作并行运算的线程数
        intra_op_parallelism_threads=num_cpu)  # 多个操作并行运算的线程数
    return tf.Session(config=tf_config)  # 用该配置运行session


def single_threaded_session():  # 单线程配置的会话
    """Returns a session which will only use a single CPU"""
    return make_session(1)


ALREADY_INITIALIZED = set()  # ALREADY_INITIALIZED是一个无序不重复集合，重复被删除


def initialize():  # 初始化新的tf变量，防止重复初始化变量
    """Initialize all the uninitialized variables in the global scope."""
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


# ================================================================
# Scopes
# ================================================================


def scope_vars(scope, trainable_only=False):  # 获得一个scope内的全部变量或可以被训练的全部变量
    """
    Get variables inside a scope
    The scope can be specified as a string

    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.

    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )  # tf.GraphKeys有全部的变量集合，get_collection函数收集有key特征的变量，并且可以指定命名空间，返回变量集合


def scope_name():
    """Returns the name of current scope as a string, e.g. deepq/q_func"""
    return tf.get_variable_scope().name  # 获取当前scope的名字


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    return scope_name() + "/" + relative_scope_name  # 多重scope时加上外层scope名字


# ================================================================
# Saving variables存储变量
# ================================================================


def load_state(fname, saver=None):
    """Load all the variables to the current session from the location <fname>"""
    if saver is None:
        saver = tf.train.Saver()
    saver.restore(get_session(), fname)  # 使用当前默认会话恢复
    return saver


def save_state(fname, saver=None):  # 将当前会话中的所有变量及参数值存储到某个路径中
    """Save all the variables in the current session to the location <fname>"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if saver is None:
        saver = tf.train.Saver()
    saver.save(get_session(), fname)
    return saver


# ================================================================
# Theano-like Function
# ================================================================

# givens即为填入inputs的数据，inputs是一批占位符，整个函数计算outputs和updates，updates中是一批可更新值的变量
# outputs是输出节点依赖于输入，updates猜测是optimizer的训练节点用于训练调整参数，整个函数完成集中使用session批量训练节点的功能
def function(inputs, outputs, updates=None, givens=None):  # important！！！！！！！！！！！！！！
    """Just like Theano function. Take a bunch of tensorflow placeholders and expersions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be feed to the inputs placeholders and produces the values of the experessions
    in outputs.从input输出到output的函数

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).

    givens输入填充入占位符placeholder的数据
    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()

            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12

    Parameters
    ----------
    inputs: [tf.placeholder or TfInput]
        list of input arguments
    outputs: [tf.Variable] or tf.Variable,变量集合，或变量
        list of outputs or a single output to be returned from function. Returned
        value will also have the same shape.
    """
    if isinstance(outputs, list):  # 根据outputs的类型调用不同的_function，是否输出一组变量，_call_函数使类的创建有返回值..
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(
            zip(outputs.keys(), f(*args, **kwargs)))  # http://www.cnblogs.com/beiluowuzheng/p/8461518.html
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]  # LAMBAD表达式简易函数，左参数右值


class _Function(object):  # 需要被function函数使用，组合训练节点
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        for inpt in inputs:
            if not issubclass(type(inpt), TfInput):  # Tfinput是最大的父类
                assert len(inpt.op.inputs) == 0, "inputs should all be placeholders of rl_algs.common.TfInput"
                # 报错，异常处理，不属于placeholder类，并且还要求输入变量，报错
        self.inputs = inputs
        updates = updates or []  # 有updates表达式就为[optimize_expr]
        self.update_group = tf.group(
            *updates)  # *降维，拆为单个元素，组合训练updates内的节点https://blog.csdn.net/LoseInVain/article/details/81703786
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens
        self.check_nan = check_nan

    def _feed_input(self, feed_dict, inpt, value):  # 对任意一个用于输入的占位符填充初值，支持placeholder型或自定类型
        if issubclass(type(inpt), TfInput):
            feed_dict.update(inpt.make_feed_dict(value))
        elif is_placeholder(inpt):
            feed_dict[inpt] = value

    # 重构调用函数.....
    def __call__(self, *args, **kwargs):  # https://www.cnblogs.com/zhangzhuozheng/p/8053045.html
        assert len(args) <= len(self.inputs), "Too many arguments provided"  # 填入的初值太多，超出需要
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):  # 依次序填入初值
            self._feed_input(feed_dict, inpt, value)
        # Update the kwargs
        kwargs_passed_inpt_names = set()  # 无顺序不重复集合
        for inpt in self.inputs[len(args):]:
            inpt_name = inpt.name.split(':')[0]
            inpt_name = inpt_name.split('/')[-1]
            assert inpt_name not in kwargs_passed_inpt_names, \
                "this function has two arguments with the same name \"{}\", so kwargs cannot be used.".format(inpt_name)
            if inpt_name in kwargs:
                kwargs_passed_inpt_names.add(inpt_name)
                self._feed_input(feed_dict, inpt, kwargs.pop(inpt_name))
            else:
                assert inpt in self.givens, "Missing argument " + inpt_name
        assert len(kwargs) == 0, "Function got extra arguments " + str(list(kwargs.keys()))
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        # 最后一步组合训练节点

        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan detected")
        return results
