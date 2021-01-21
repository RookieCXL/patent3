
from arch.api.utils import log_utils
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import DecisionTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import DecisionTreeModelParam
from federatedml.tree import DecisionTree
from federatedml.tree import Splitter
from federatedml.tree import SplitInfo
from federatedml.tree import FeatureHistogram
from federatedml.transfer_variable.transfer_class.hetero_decision_tree_transfer_variable import \
    HeteroDecisionTreeTransferVariable
from federatedml.util import consts
from federatedml.tree import Node
from federatedml.tree.hetero.hetero_decision_tree_host import HeteroDecisionTreeHost
from federatedml.feature.fate_element_type import NoneType
from arch.api.utils.splitable import segment_transfer_enabled
import functools
import threading
LOGGER = log_utils.getLogger()
class MyThread(threading.Thread):
    """
    多线程类，进行线程中参数传导和线程启用
    """
    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args
    def run(self):
        #线程启用
        self.result = self.func(*self.args)
    def get_result(self):
        """
        线程输出结果
        :return result
        """
        try:
            return self.result
        except Exception:
            return None


class MyHeteroDecisionTreeHost(HeteroDecisionTreeHost):
    def __init__(self, tree_param):
        """
        单颗决策树的推理
        :param tree_param:
        """
        LOGGER.info("hetero decision tree guest init!")
        super(MyHeteroDecisionTreeHost, self).__init__(tree_param)


    def sync_weight_dict(self, recv_times):
        """
        接收业务方数据信息
        :param recv_times:
        :return:
        """
        LOGGER.info("srecv weight dict to host, recv times is {}".format(recv_times))
        LOGGER.info("cxl:sync_weight_sict_start")
        weight_dict = self.transfer_inst.weight_dict.get(idx=0,
                                                        suffix=(recv_times,))
        LOGGER.info("cxl:sync_weight_sict_end")
        return weight_dict

    def sync_predict_result_by_host(self, predict_result_by_host, send_times):
        """
        本地推理数据发送业务方
        :param predict_result_by_host:
        :param send_times:
        :return:
        """
        LOGGER.info("send weight dict by host, send times is {}".format(send_times))

        self.transfer_inst.predict_result_by_host.remote(predict_result_by_host,
                                                       role=consts.GUEST,
                                                       idx=0,
                                                       suffix=(send_times,))


    @staticmethod
    def traverse_tree(predict_state, data_inst, tree_=None,
                      decoder=None, sitename=consts.GUEST, split_maskdict=None,
                      use_missing=None, zero_as_missing=None, missing_dir_maskdict=None,
                      encrypted_weight_dict=None, encrypted_zero_dict=None):
        """
        树的广度优先遍历过程，在搜索过程中，将节点是否可行进行one-hot编码
        :param predict_state:
        :param data_inst:
        :param tree_:
        :param decoder:
        :param sitename:
        :param split_maskdict:
        :param use_missing:
        :param zero_as_missing:
        :param missing_dir_maskdict:
        :param encrypted_weight_dict:
        :param encrypted_zero_dict:
        :return weight_dict
        """
        nid, tag = predict_state

        weight_dict = {}
        #Encrypt(0)
        # LOGGER.info("cxl:tree_{}".format(tree_))
        # LOGGER.info("CXL:len(tree_)".format(len(tree_)))
        # for i in range(len(tree_)):
        #     # LOGGER.info("cxl:i{}".format(i))
        #     if tree_[i].is_leaf is True:
        #         weight_dict[i] = encrypted_zero_dict[i]

        node_queue = [nid]

        while len(node_queue) != 0:
            nid = node_queue[0]
            node_queue.remove(nid)
            #leaf_code weight
            if tree_[nid].is_leaf is True:
                weight_dict[nid] = encrypted_weight_dict[nid]
            else:
                if tree_[nid].sitename == sitename:
                    fid = decoder("feature_idx", tree_[nid].fid, split_maskdict=split_maskdict)
                    bid = decoder("feature_val", tree_[nid].bid, nid, split_maskdict=split_maskdict)

                    if not use_missing:
                        missing_dir = 1
                    else:
                        missing_dir = decoder("missing_dir", 1, nid, missing_dir_maskdict=missing_dir_maskdict)

                        if  zero_as_missing:
                            missing_dir = decoder("missing_dir", 1, nid, missing_dir_maskdict=missing_dir_maskdict)
                            if data_inst.features.get_data(fid) == NoneType() or data_inst.features.get_data(fid,
                                                                                                             None) is None:
                                if missing_dir == 1:
                                    nid = tree_[nid].right_nodeid
                                else:
                                    nid = tree_[nid].left_nodeid
                            elif data_inst.features.get_data(fid) <= bid:
                                nid = tree_[nid].left_nodeid
                            else:
                                nid = tree_[nid].right_nodeid
                        elif data_inst.features.get_data(fid) == NoneType():
                            if missing_dir == 1:
                                nid = tree_[nid].right_nodeid
                            else:
                                nid = tree_[nid].left_nodeid
                        elif data_inst.features.get_data(fid, 0) <= bid:
                            nid = tree_[nid].left_nodeid
                        else:
                            nid = tree_[nid].right_nodeid
                        node_queue.append(nid)
                else:
                    node_queue.append(tree_[nid].left_nodeid)
                    node_queue.append(tree_[nid].right_nodeid)

        return weight_dict

    @staticmethod
    def traversal(result_weight_dict_record,predict_data_weight_dict):
        """
        将数据方和业务方的结果进行合并
        :return result_weight
        """
        result_weight = 0
        for keys in result_weight_dict_record:
            if(result_weight_dict_record[keys]==1):
                result_weight+=predict_data_weight_dict[keys]
        return result_weight


    def predict(self, data_inst, result_weight_dict_record=None, predict_data_weight_dict=None):
        """
        数据方树的遍历和接收业务方参数异步进行，并进行参数合并
        :return predict_result_by_host
        """
        LOGGER.info("cxl decision tree start to predict!")
        def my_traversal(traverse_tree,data_inst,predict_data):
            """
            数据方树的遍历
            :return result_weight_dict_record
            """
            #数据方树的遍历
            result_weight_dict_record = predict_data.join(data_inst, traverse_tree)
            # LOGGER.info("csl:result_weight_dict_record:{}".format(list(result_weight_dict_record.collect())))
            return result_weight_dict_record

        def my_sync_weight_dict_data():
            """
            接收业务方参数
            :return predict_data_weight_dict
            """
            predict_data_weight_dict = self.sync_weight_dict(0)
            # LOGGER.info("csl:predict_data_weight_dict:{}".format(list(predict_data_weight_dict.collect())))
            return predict_data_weight_dict

        predict_data = data_inst.mapValues(lambda data_inst: (0, 1))
        encrypted_weight_dict = {}
        encrypted_zero_dict = {}
        for i in range(len(self.tree_)):
            cur_node = self.tree_[i]
            if cur_node.is_leaf:
                # encrypted_weight_dict[cur_node.id] = cur_node.weight
                # encrypted_zero_dict[cur_node.id] = 0
                encrypted_weight_dict[cur_node.id] = 1
                encrypted_zero_dict[cur_node.id] = 0
        traverse_tree = functools.partial(self.traverse_tree,
                                          tree_=self.tree_,
                                          decoder=self.decode,
                                          sitename=self.sitename,
                                          split_maskdict=self.split_maskdict,
                                          use_missing=self.use_missing,
                                          zero_as_missing=self.zero_as_missing,
                                          missing_dir_maskdict=self.missing_dir_maskdict,
                                          encrypted_weight_dict=encrypted_weight_dict,
                                          encrypted_zero_dict=encrypted_zero_dict)

        threads=[]
        t1 =MyThread(my_sync_weight_dict_data)
        threads.append(t1)
        t2=MyThread(my_traversal,args=(traverse_tree,data_inst,predict_data))
        threads.append(t2)
        LOGGER.info("cxl:traversal and sync start!")

        #树的搜索和与业务方信息交互异步进行
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        LOGGER.info("cxl:traversal and sync end!")
        result_weight_dict_record=t2.get_result()
        predict_data_weight_dict=t1.get_result()
        LOGGER.info("cxl:combine the traversal  start!")
        predict_result_by_host = result_weight_dict_record.join(predict_data_weight_dict, lambda v1, v2: sum([v2[i] for i in v1.keys()]))
        LOGGER.info("cxl:combine the traversal  end!")
        LOGGER.info("cxl decision tree predict end!")

        return predict_result_by_host
