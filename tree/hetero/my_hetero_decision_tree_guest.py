
import copy
import functools

from arch.api import session
from arch.api.utils import log_utils
from federatedml.feature.fate_element_type import NoneType
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import CriterionMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import DecisionTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import DecisionTreeModelParam
from federatedml.transfer_variable.transfer_class.hetero_decision_tree_transfer_variable import \
    HeteroDecisionTreeTransferVariable
from federatedml.tree import DecisionTree
from federatedml.tree import FeatureHistogram
from federatedml.tree import Node
from federatedml.tree.hetero.hetero_decision_tree_guest import HeteroDecisionTreeGuest
from federatedml.tree import Splitter
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class MyHeteroDecisionTreeGuest(HeteroDecisionTreeGuest):
    def __init__(self, tree_param):
        LOGGER.info("my hetero decision tree guest init!")
        super(MyHeteroDecisionTreeGuest, self).__init__(tree_param)
        self.splitter = Splitter(self.criterion_method, self.criterion_params, self.min_impurity_split,
                                 self.min_sample_split, self.min_leaf_node)

        self.data_bin = None
        self.grad_and_hess = None
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.data_bin_with_node_dispatch = None
        self.node_dispatch = None
        self.infos = None
        self.valid_features = None
        self.encrypter = None
        self.encrypted_mode_calculator = None
        self.best_splitinfo_guest = None
        self.tree_node_queue = None
        self.cur_split_nodes = None
        self.tree_ = []
        self.tree_node_num = 0
        self.split_maskdict = {}
        self.missing_dir_maskdict = {}
        self.transfer_inst = HeteroDecisionTreeTransferVariable()
        self.predict_weights = None
        self.host_party_idlist = []
        self.runtime_idx = 0
        self.sitename = consts.GUEST
        self.feature_importances_ = {}
    def sync_weight_dict(self, weight_dict, send_times):
        LOGGER.info("send weight data to host, sending times is {}".format(send_times))
        LOGGER.info("cxl:send weight data to host start!")
        self.transfer_inst.weight_dict.remote(weight_dict,
                                               role=consts.HOST,
                                               idx=-1,
                                               suffix=(send_times,))
        LOGGER.info("cxl:send weight data to host end!")

    def sync_predict_result_by_host(self, send_times):
        LOGGER.info("get predict result by host by host, recv times is {}".format(send_times))
        predict_result_by_host = self.transfer_inst.predict_result_by_host.get(idx=-1,
                                                                   suffix=(send_times,))
        return predict_result_by_host

    @staticmethod
    def traverse_tree(predict_state, data_inst, tree_=None,
                      decoder=None, sitename=consts.GUEST, split_maskdict=None,
                      use_missing=None, zero_as_missing=None, missing_dir_maskdict=None,
                      encrypted_weight_dict=None, encrypted_zero_dict=None):
        nid, tag = predict_state

        weight_dict = {}
        #Encrypt(0)
        # LOGGER.info("cxl:tree_{}".format(tree_))
        # LOGGER.info("CXL:len(tree_)".format(len(tree_)))
        for i in range(len(tree_)):
            # LOGGER.info("cxl:i{}".format(i))
            if tree_[i].is_leaf is True:
                weight_dict[i] = encrypted_zero_dict[i]

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

                    if use_missing:
                        missing_dir = decoder("missing_dir", 1, nid, missing_dir_maskdict=missing_dir_maskdict)
                    else:
                        missing_dir = 1

                    if use_missing and zero_as_missing:
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

    def predict(self, data_inst):
        LOGGER.info("cxl decision tree start to predict!")
        predict_data = data_inst.mapValues(lambda data_inst: (0, 1))

        LOGGER.debug("cxl start to encrypt!")
        encrypted_weight_dict = {}
        encrypted_zero_dict = {}
        for i in range(len(self.tree_)):
            cur_node = self.tree_[i]
            if cur_node.is_leaf:
                encrypted_weight_dict[cur_node.id] = cur_node.weight
                encrypted_zero_dict[cur_node.id] = 0
                #encrypted_weight_dict[cur_node.id] = self.encrypt(cur_node.weight)
                #encrypted_zero_dict[cur_node.id] = self.encrypt(0)
        LOGGER.debug("cxl encrypt end!")

        LOGGER.info("cxl decision tree start to traverse!")

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
        predict_result_weight_dict = predict_data.join(data_inst, traverse_tree)
        LOGGER.info("cxl decision tree traverse end!")

        self.sync_weight_dict(predict_result_weight_dict, 0)
        LOGGER.info("cxl decision tree predict end!")

        # predict_result = self.sync_predict_result_by_host(0)
        #
        # LOGGER.info("cxl predict finish!")
        #
        # LOGGER.debug("cxl start to decrypt!")
        # # for only one host
        # predict_result = predict_result[0].mapValues(lambda enc_weight: self.decrypt(enc_weight))
        # # predict_result = predict_result[0].mapValues(lambda enc_weight: enc_weight)
        # LOGGER.debug("cxl decrypt end!")

        # return predict_result
