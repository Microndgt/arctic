import logging
import numpy as np
import numpy.ma as ma
import pandas as pd

try:
    from pandas.api.types import infer_dtype
except ImportError:
    from pandas.lib import infer_dtype

try:
    from pandas._libs.lib import max_len_string_array
except ImportError:
    from pandas.lib import max_len_string_array

from bson import Binary, SON, BSON

from .._compression import compress, decompress, compress_array
from ._serializer import Serializer



DATA = 'd'
MASK = 'm'
TYPE = 't'
DTYPE = 'dt'
COLUMNS = 'c'
INDEX = 'i'
METADATA = 'md'
LENGTHS = 'ln'


class FrameConverter(object):
    """
    Converts a Pandas Dataframe to and from PyMongo SON representation:

        {
          METADATA: {
                      COLUMNS: [col1, col2, ...]             list of str
                      MASKS: {col1: mask, col2: mask, ...}   dict of str: Binary
                      INDEX: [idx1, idx2, ...]               list of str
                      TYPE: 'series' or 'dataframe'
                      LENGTHS: {col1: len, col2: len, ...}   dict of str: int
                    }
          DATA: BINARY(....)      Compressed columns concatenated together
        }
    """

    def _convert_types(self, a):
        """
        Converts object arrays of strings to numpy string arrays
        """
        # No conversion for scalar type
        # dtype('O')
        if a.dtype != 'object':
            return a, None

        # We can't infer the type of an empty array, so just
        # assume strings
        if len(a) == 0:
            return a.astype('U1'), None

        # Compute a mask of missing values. Replace NaNs and Nones with
        # empty strings so that type inference has a chance.
        # 判断a数组中是否有NULL值
        mask = pd.isnull(a)
        # 如果有NULL值，
        if mask.sum() > 0:
            a = a.copy()
            # a的大小要和mask对应，这样mask为True对应a的值，将会被替换为 ''
            np.putmask(a, mask, '')
        else:
            mask = None

        if infer_dtype(a) == 'mixed':
            # assume its a string, otherwise raise an error
            try:
                a = np.array([s.encode('ascii') for s in a])
                a = a.astype('O')
            except:
                raise ValueError("Column of type 'mixed' cannot be converted to string")

        type_ = infer_dtype(a)
        if type_ in ['unicode', 'string']:
            # 最大长度
            max_len = max_len_string_array(a)
            # 将其转换成指定的类型
            return a.astype('U{:d}'.format(max_len)), mask
        else:
            raise ValueError('Cannot store arrays with {} dtype'.format(type_))

    def docify(self, df):
        """
        Convert a Pandas DataFrame to SON.

        Parameters
        ----------
        df:  DataFrame
            The Pandas DataFrame to encode
        """
        dtypes = {}
        masks = {}
        lengths = {}
        columns = []
        data = Binary(b'')
        start = 0

        arrays = []
        for c in df:
            # 迭代的是列名
            try:
                # df[c]则获取该列的所有值
                columns.append(str(c))
                # df[c].values将其转换成一个array对象, numpy.ndarray
                # 这个函数的作用就是，推断df[c]的类型和长度，以及是否为None，传回来一个mask
                # 并且将数据转换成指定的类型，
                # 比如对于SYMBOL列，array(['600000_CNSESH', '600004_CNSESH', '600005_CNSESH', ...,
                #  '300586_CNSESZ', '300587_CNSESZ', '300588_CNSESZ'],
                # dtype='<U13')
                # mask是一个True，False的数组，或者是一个None
                arr, mask = self._convert_types(df[c].values)
                # 记录列名对应的类型信息
                dtypes[str(c)] = arr.dtype.str
                if mask is not None:
                    # 记录NULL值信息
                    masks[str(c)] = Binary(compress(mask.tostring()))
                arrays.append(arr.tostring())
            except Exception as e:
                typ = infer_dtype(df[c])
                msg = "Column '{}' type is {}".format(str(c), typ)
                logging.info(msg)
                raise e
        # arrays是包含所有列的数据字符串
        arrays = compress_array(arrays)
        for index, c in enumerate(df):
            # 将每列的字符串数据转换成Binary
            d = Binary(arrays[index])
            # 记录长度信息
            lengths[str(c)] = (start, start + len(d) - 1)
            start += len(d)
            data += d

        doc = SON({DATA: data, METADATA: {}})
        doc[METADATA] = {COLUMNS: columns,
                         MASK: masks,
                         LENGTHS: lengths,
                         DTYPE: dtypes
                         }

        return doc

    def objify(self, doc, columns=None):
        """
        Decode a Pymongo SON object into an Pandas DataFrame
        """
        cols = columns or doc[METADATA][COLUMNS]
        data = {}

        for col in cols:
            d = decompress(doc[DATA][doc[METADATA][LENGTHS][col][0]: doc[METADATA][LENGTHS][col][1] + 1])
            d = np.fromstring(d, doc[METADATA][DTYPE][col])

            if MASK in doc[METADATA] and col in doc[METADATA][MASK]:
                mask_data = decompress(doc[METADATA][MASK][col])
                mask = np.fromstring(mask_data, 'bool')
                d = ma.masked_array(d, mask)
            data[col] = d

        return pd.DataFrame(data, columns=cols)[cols]


class FastFrameConverter(FrameConverter):
    def docify(self, df):
        """
        将Pandas的DataFrame转换成列表字典

        :param df:
        :return:
        """
        data = df.to_dict(orient='records')
        return SON({DATA: data, METADATA: {}})

    def objify(self, doc, columns=None):
        """
        将一个SON对象转换成Pandas的DataFrame

        :param doc:
        :param columns:
        :return:
        """
        return pd.DataFrame(doc[DATA])


class FrametoArraySerializer(Serializer):
    TYPE = 'FrameToArray'

    def __init__(self, converter=None):
        self.converter = converter or FrameConverter()

    def serialize(self, df):
        if isinstance(df, pd.Series):
            dtype = 'series'
            df = df.to_frame()
        else:
            dtype = 'dataframe'

        if (len(df.index.names) > 1 and None in df.index.names) or None in list(df.columns.values):
            raise Exception("All columns and indexes must be named")

        # 处理有关index的东西
        if df.index.names != [None]:
            index = df.index.names
            # 将index值转换成一列，然后创建一个简单的数字索引
            # 也就是说date必须是df的一列
            df = df.reset_index()
            # 开始序列化
            ret = self.converter.docify(df)
            # 所以这里元信息要加上index信息
            ret[METADATA][INDEX] = index
            ret[METADATA][TYPE] = dtype
            return ret
        ret = self.converter.docify(df)
        ret[METADATA][TYPE] = dtype
        return ret

    def deserialize(self, data, columns=None):
        '''
        Deserializes SON to a DataFrame

        Parameters
        ----------
        data: SON data
        columns: None, or list of strings
            optionally you can deserialize a subset of the data in the SON. Index
            columns are ALWAYS deserialized, and should not be specified

        Returns
        -------
        pandas dataframe or series
        '''
        if data == []:
            return pd.DataFrame()

        meta = data[0][METADATA] if isinstance(data, list) else data[METADATA]
        index = INDEX in meta

        if columns:
            if index:
                columns.extend(meta[INDEX])
            if len(columns) > len(set(columns)):
                raise Exception("Duplicate columns specified, cannot de-serialize")

        if not isinstance(data, list):
            df = self.converter.objify(data, columns)
        else:
            df = pd.concat([self.converter.objify(d, columns) for d in data], ignore_index=not index)

        if index:
            df = df.set_index(meta[INDEX])
        if meta[TYPE] == 'series':
            return df[df.columns[0]]
        return df

    def combine(self, a, b):
        if a.index.names != [None]:
            return pd.concat([a, b]).sort_index()
        return pd.concat([a, b])
